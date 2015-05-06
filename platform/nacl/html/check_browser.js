/*
 * Copyright (c) 2011 The Native Client Authors. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

/**
 * @fileoverview This file provides a BrowserChecker Javascript class.
 * Users can create a BrowserChecker object, invoke checkBrowser(|version|),
 * and then use getIsValidBrowser() and getBrowserSupportStatus()
 * to determine if the browser version is greater than |version|
 * and if the Native Client plugin is found.
 */

// Create a namespace object
var browser_version = browser_version || {};

/**
 * Class to provide checking for version and NativeClient.
 * @param {integer} arg1 An argument that indicates major version of Chrome we
 *     require, such as 14.
 */

/**
 * Constructor for the BrowserChecker.  Sets the major version of
 * Chrome that is required to |minChromeVersion|.
 * @param minChromeVersion   The earliest major version of chrome that
 *     is supported.  If the Chrome browser version is less than
 *     |minChromeVersion| then |isValidBrowswer| will be set to false.
 * @param opt_maxChromeVersion   Ignored.  Retained for backwards compatibility.
 * @param appVersion  The application version string.
 * @param plugins     The plugins that exist in the browser.
 * @constructor
 */
browser_version.BrowserChecker = function(minChromeVersion,
                                          appVersion, plugins,
                                          opt_maxChromeVersion) {
  /**
   * Version specified by the user. This class looks to see if the browser
   * version is >= |minChromeVersion_|.
   * @type {integer}
   * @private
   */
  this.minChromeVersion_ = minChromeVersion;

  /**
   * List of Browser plugin objects.
   * @type {Ojbect array}
   * @private
   */
  this.plugins_ = plugins;

  /**
   * Application version string from the Browser.
   * @type {integer}
   * @private
   */
  this.appVersion_ = appVersion;

  /**
   * Flag used to indicate if the browser has Native Client and is if the
   * browser version is recent enough.
   * @type {boolean}
   * @private
   */
  this.isValidBrowser_ = false;

  /**
   * Actual major version of Chrome -- found by querying the browser.
   * @type {integer}
   * @private
   */
  this.chromeVersion_ = null;

  /**
   * Browser support status. This allows the user to get a detailed status
   * rather than using this.browserSupportMessage.
   */
  this.browserSupportStatus_ =
      browser_version.BrowserChecker.StatusValues.UNKNOWN;
}

/**
 * The values used for BrowserChecker status to indicate success or
 * a specific error.
 * @enum {id}
 */
browser_version.BrowserChecker.StatusValues = {
  UNKNOWN: 0,
  NACL_ENABLED: 1,
  UNKNOWN_BROWSER: 2,
  CHROME_VERSION_TOO_OLD: 3,
  NACL_NOT_ENABLED: 4,
  NOT_USING_SERVER: 5
};

/**
 * Determines if the plugin with name |name| exists in the browser.
 * @param {string} name The name of the plugin.
 * @param {Object array} plugins The plugins in this browser.
 * @return {bool} |true| if the plugin is found.
 */
browser_version.BrowserChecker.prototype.pluginExists = function(name,
                                                                 plugins) {
  for (var index=0; index < plugins.length; index++) {
    var plugin = this.plugins_[index];
    var plugin_name = plugin['name'];
    // If the plugin is not found, you can use the Javascript console
    // to see the names of the plugins that were found when debugging.
    if (plugin_name.indexOf(name) != -1) {
      return true;
    }
  }
  return false;
}

/**
 * Returns browserSupportStatus_ which indicates if the browser supports
 * Native Client.  Values are defined as literals in
 * browser_version.BrowserChecker.StatusValues.
 * @ return {int} Level of NaCl support.
 */
browser_version.BrowserChecker.prototype.getBrowserSupportStatus = function() {
  return this.browserSupportStatus_;
}

/**
 * Returns isValidBrowser (true/false) to indicate if the browser supports
 * Native Client.
 * @ return {bool} If this browser has NativeClient and correct version.
 */
browser_version.BrowserChecker.prototype.getIsValidBrowser = function() {
  return this.isValidBrowser_;
}

/**
 * Checks to see if this browser can support Native Client applications.
 * For Chrome browsers, checks to see if the "Native Client" plugin is
 * enabled.
 */
browser_version.BrowserChecker.prototype.checkBrowser = function() {
  var versionPatt = /Chrome\/(\d+)\.(\d+)\.(\d+)\.(\d+)/;
  var result = this.appVersion_.match(versionPatt);

  // |result| stores the Chrome version number.
  if (!result) {
    this.isValidBrowser_ = false;
    this.browserSupportStatus_ =
        browser_version.BrowserChecker.StatusValues.UNKNOWN_BROWSER;
  } else {
    this.chromeVersion_ = result[1];
    // We know we have Chrome, check version and/or plugin named Native Client
    if (this.chromeVersion_ >= this.minChromeVersion_) {
      var found_nacl = this.pluginExists('Native Client', this.plugins_);
      if (found_nacl) {
        this.isValidBrowser_ = true;
        this.browserSupportStatus_ =
            browser_version.BrowserChecker.StatusValues.NACL_ENABLED;
      } else {
        this.isValidBrowser_ = false;
        this.browserSupportStatus_ =
            browser_version.BrowserChecker.StatusValues.NACL_NOT_ENABLED;
      }
    } else {
      // We are in a version that is less than |minChromeVersion_|
      this.isValidBrowser_ = false;
      this.browserSupportStatus_ =
          browser_version.BrowserChecker.StatusValues.CHROME_VERSION_TOO_OLD;
    }
  }
  var my_protocol = window.location.protocol;
  if (my_protocol.indexOf('file') == 0) {
    this.isValidBrowser_ = false;
    this.browserSupportStatus_ =
        browser_version.BrowserChecker.StatusValues.NOT_USING_SERVER;
  }
}

