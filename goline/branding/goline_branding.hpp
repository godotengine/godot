/**************************************************************************/
/*  goline_branding.hpp                                                   */
/**************************************************************************/
/*   GOLINE — Goline-specific branding constants.                          */
/*                                                                        */
/*   This header is Goline-specific and lives under goline/ so it can     */
/*   never be confused with upstream Godot code. It is purely additive:   */
/*   NO upstream Godot file is modified to include it yet.                */
/*                                                                        */
/*   It is the single source of truth for Goline's identity strings.      */
/*   Later stages may consume these constants when Goline visibly          */
/*   re-brands the engine (see docs/goline/IDENTITY.md).                  */
/**************************************************************************/

#ifndef GOLINE_BRANDING_HPP
#define GOLINE_BRANDING_HPP

#define GOLINE_SHORT_NAME "goline"
#define GOLINE_NAME "Goline"
#define GOLINE_DISPLAY_NAME "Goline"
#define GOLINE_WEBSITE ""

// Relationship to upstream: Goline is a fork of the Godot Engine.
// Keep a clear reference to the upstream project it is derived from.
#define GOLINE_BASED_ON "Godot Engine"
#define GOLINE_BASED_ON_WEBSITE "https://godotengine.org"

// Default display label, e.g. for window titles and the About dialog.
#define GOLINE_VERSION_LABEL GOLINE_DISPLAY_NAME

#endif // GOLINE_BRANDING_HPP
