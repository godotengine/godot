// clang-format off

/* Third-party library.
 * Upstream: https://github.com/JakeWharton/ProcessPhoenix
 * Commit: 12cb27c2cc9c3fc555e97f2db89e571667de82c4
 */

/*
 * Copyright (C) 2014 Jake Wharton
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.godotengine.godot.utils;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.ActivityOptions;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Process;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static android.content.Intent.FLAG_ACTIVITY_CLEAR_TASK;
import static android.content.Intent.FLAG_ACTIVITY_NEW_TASK;

/**
 * Process Phoenix facilitates restarting your application process. This should only be used for
 * things like fundamental state changes in your debug builds (e.g., changing from staging to
 * production).
 * <p>
 * Trigger process recreation by calling {@link #triggerRebirth} with a {@link Context} instance.
 */
public final class ProcessPhoenix extends Activity {
  private static final String KEY_RESTART_INTENTS = "phoenix_restart_intents";
  // -- GODOT start --
  private static final String KEY_RESTART_ACTIVITY_OPTIONS = "phoenix_restart_activity_options";
  // -- GODOT end --
  private static final String KEY_MAIN_PROCESS_PID = "phoenix_main_process_pid";

  /**
   * Call to restart the application process using the {@linkplain Intent#CATEGORY_DEFAULT default}
   * activity as an intent.
   * <p>
   * Behavior of the current process after invoking this method is undefined.
   */
  public static void triggerRebirth(Context context) {
    triggerRebirth(context, getRestartIntent(context));
  }

  // -- GODOT start --
  /**
   * Call to restart the application process using the specified intents.
   * <p>
   * Behavior of the current process after invoking this method is undefined.
   */
  public static void triggerRebirth(Context context, Intent... nextIntents) {
    triggerRebirth(context, null, nextIntents);
  }

  /**
   * Call to restart the application process using the specified intents launched with the given
   * {@link ActivityOptions}.
   * <p>
   * Behavior of the current process after invoking this method is undefined.
   */
  public static void triggerRebirth(Context context, Bundle activityOptions, Intent... nextIntents) {
    if (nextIntents.length < 1) {
      throw new IllegalArgumentException("intents cannot be empty");
    }
    // create a new task for the first activity.
    nextIntents[0].addFlags(FLAG_ACTIVITY_NEW_TASK | FLAG_ACTIVITY_CLEAR_TASK);

    Intent intent = new Intent(context, ProcessPhoenix.class);
    intent.addFlags(FLAG_ACTIVITY_NEW_TASK); // In case we are called with non-Activity context.
    intent.putParcelableArrayListExtra(KEY_RESTART_INTENTS, new ArrayList<>(Arrays.asList(nextIntents)));
    intent.putExtra(KEY_MAIN_PROCESS_PID, Process.myPid());
    if (activityOptions != null) {
      intent.putExtra(KEY_RESTART_ACTIVITY_OPTIONS, activityOptions);
    }
    context.startActivity(intent);
  }

  /**
   * Finish the activity and kill its process
   */
  public static void forceQuit(Activity activity) {
    forceQuit(activity, Process.myPid());
  }

  /**
   * Finish the activity and kill its process
   * @param activity
   * @param pid
   */
  public static void forceQuit(Activity activity, int pid) {
    Process.killProcess(pid); // Kill original main process
    activity.finishAndRemoveTask();
    Runtime.getRuntime().exit(0); // Kill kill kill!
  }

  // -- GODOT end --

  private static Intent getRestartIntent(Context context) {
    String packageName = context.getPackageName();
    Intent defaultIntent = context.getPackageManager().getLaunchIntentForPackage(packageName);
    if (defaultIntent != null) {
      return defaultIntent;
    }

    throw new IllegalStateException("Unable to determine default activity for "
        + packageName
        + ". Does an activity specify the DEFAULT category in its intent filter?");
  }

  @Override protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    // -- GODOT start --
    Intent launchIntent = getIntent();
    ArrayList<Intent> intents = launchIntent.getParcelableArrayListExtra(KEY_RESTART_INTENTS);
    Bundle activityOptions = launchIntent.getBundleExtra(KEY_RESTART_ACTIVITY_OPTIONS);
    startActivities(intents.toArray(new Intent[intents.size()]), activityOptions);
    forceQuit(this, launchIntent.getIntExtra(KEY_MAIN_PROCESS_PID, -1));
    // -- GODOT end --
  }

  /**
   * Checks if the current process is a temporary Phoenix Process.
   * This can be used to avoid initialization of unused resources or to prevent running code that
   * is not multi-process ready.
   *
   * @return true if the current process is a temporary Phoenix Process
   */
  public static boolean isPhoenixProcess(Context context) {
    int currentPid = Process.myPid();
    ActivityManager manager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
    List<ActivityManager.RunningAppProcessInfo> runningProcesses = manager.getRunningAppProcesses();
    if (runningProcesses != null) {
      for (ActivityManager.RunningAppProcessInfo processInfo : runningProcesses) {
        if (processInfo.pid == currentPid && processInfo.processName.endsWith(":phoenix")) {
          return true;
        }
      }
    }
    return false;
  }
}
