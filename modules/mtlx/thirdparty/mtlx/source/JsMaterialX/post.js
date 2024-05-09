//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//
var postRegistrations = [];

function onModuleReady(callback) {
    postRegistrations.push(callback);
}

// This callback should only be registered once in all post JS scripts.
// That's why we do it globally here, and let other scripts register their code via 'onModuleReady'.
Module.onRuntimeInitialized = function() {
    for (var callback of postRegistrations) {
        callback();
    }
};