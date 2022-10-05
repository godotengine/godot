package org.godotengine.editor

/**
 * Describe the editor instance to launch
 */
data class EditorInstanceInfo(
	val instanceClassName: String,
	val instanceId: Int,
	val processNameSuffix: String,
	val launchAdjacent: Boolean = false
) {
	constructor(
			instanceClass: Class<*>,
			instanceId: Int,
			processNameSuffix: String,
			launchAdjacent: Boolean = false
	) : this(instanceClass.name, instanceId, processNameSuffix, launchAdjacent)
}
