/**
 * @file gaussian_splat_debug_hud.h
 * @brief In-viewport debug HUD for Gaussian Splatting performance stats.
 *
 * This file defines GaussianSplatDebugHUD, a Control node that draws runtime
 * statistics directly in the viewport. It displays splat count, FPS, memory
 * usage, streaming status, and other key performance metrics.
 */

#ifndef GAUSSIAN_SPLAT_DEBUG_HUD_H
#define GAUSSIAN_SPLAT_DEBUG_HUD_H

#include "scene/gui/control.h"
#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"

class GaussianSplatNode3D;
class Font;

/**
 * @class GaussianSplatDebugHUD
 * @brief Control node that renders debug statistics overlay in the viewport.
 *
 * GaussianSplatDebugHUD is a lightweight UI control that displays key rendering
 * statistics for Gaussian Splatting directly in the game/editor viewport. It is
 * automatically created and managed by GaussianSplatNode3D when the debug HUD
 * is enabled.
 *
 * ## Displayed Information
 *
 * - **Splat Count**: Visible / Total splats
 * - **FPS / Frame Time**: Current rendering performance
 * - **GPU Memory**: Estimated VRAM usage in MB
 * - **Streaming Status**: Chunks loaded / total (if streaming enabled)
 * - **Sort/Render Time**: GPU timing for key pipeline stages
 *
 * ## Usage
 *
 * The HUD is controlled via GaussianSplatNode3D properties:
 * @code
 * splat_node.set_debug_hud_enabled(true)  # Show the HUD
 * splat_node.set_debug_hud_corner(GaussianSplatDebugHUD.CORNER_TOP_LEFT)
 * @endcode
 *
 * @note This is an internal implementation class. Use the GaussianSplatNode3D
 *       interface to control the debug HUD visibility and positioning.
 */
class GaussianSplatDebugHUD : public Control {
	GDCLASS(GaussianSplatDebugHUD, Control);

public:
	/**
	 * @enum Corner
	 * @brief Specifies which corner of the viewport the HUD appears in.
	 */
	enum Corner {
		CORNER_TOP_LEFT = 0,
		CORNER_TOP_RIGHT = 1,
		CORNER_BOTTOM_LEFT = 2,
		CORNER_BOTTOM_RIGHT = 3
	};

private:
	GaussianSplatNode3D *splat_node = nullptr;
	Corner corner = CORNER_TOP_LEFT;
	float update_interval = 0.1f;
	float time_since_update = 0.0f;

	// Cached renderer-provided HUD text (updated periodically)
	Vector<String> cached_hud_lines;
	LocalVector<Vector2> line_size_scratch;

	// Drawing resources
	Ref<Font> hud_font;
	int font_size = 14;
	Color background_color = Color(0.0f, 0.0f, 0.0f, 0.7f);
	Color text_color = Color(0.9f, 0.9f, 0.9f, 1.0f);
	Color highlight_color = Color(0.4f, 1.0f, 0.6f, 1.0f);
	Color warning_color = Color(1.0f, 0.8f, 0.2f, 1.0f);
	float margin = 10.0f;
	float padding = 8.0f;
	float line_spacing = 4.0f;

	void _update_cached_stats();
	void _draw_hud();
	Vector2 _calculate_hud_position(const Vector2 &p_hud_size) const;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	GaussianSplatDebugHUD();
	~GaussianSplatDebugHUD();

	/**
	 * @brief Links this HUD to a GaussianSplatNode3D for stats retrieval.
	 * @param p_node The splat node to monitor (may be nullptr to disconnect).
	 */
	void set_splat_node(GaussianSplatNode3D *p_node);

	/**
	 * @brief Returns the linked GaussianSplatNode3D.
	 */
	GaussianSplatNode3D *get_splat_node() const { return splat_node; }

	/**
	 * @brief Sets which corner of the viewport the HUD appears in.
	 * @param p_corner One of CORNER_TOP_LEFT, CORNER_TOP_RIGHT, etc.
	 */
	void set_corner(Corner p_corner);

	/**
	 * @brief Returns the current corner setting.
	 */
	Corner get_corner() const { return corner; }

	/**
	 * @brief Sets how often the HUD refreshes its statistics (in seconds).
	 * @param p_interval Update interval (default 0.1 = 10 Hz).
	 */
	void set_update_interval(float p_interval);

	/**
	 * @brief Returns the current update interval.
	 */
	float get_update_interval() const { return update_interval; }

	/**
	 * @brief Sets the font size for HUD text.
	 * @param p_size Font size in pixels.
	 */
	void set_font_size(int p_size);

	/**
	 * @brief Returns the current font size.
	 */
	int get_font_size() const { return font_size; }

	/**
	 * @brief Sets the background color of the HUD panel.
	 * @param p_color Background color with alpha for transparency.
	 */
	void set_background_color(const Color &p_color);

	/**
	 * @brief Returns the current background color.
	 */
	Color get_background_color() const { return background_color; }

	/**
	 * @brief Forces an immediate refresh of the displayed statistics.
	 */
	void refresh_stats();
};

VARIANT_ENUM_CAST(GaussianSplatDebugHUD::Corner);

#endif // GAUSSIAN_SPLAT_DEBUG_HUD_H
