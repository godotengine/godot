#!/usr/bin/env python
# -*- coding: utf-8 -*-
from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout
from conan.tools.build import can_run
from conan.tools.files import save, load
import os


class TestPackageConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps", "VirtualRunEnv"
    test_type = "explicit"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        save(self, os.path.join(self.build_folder, "package_folder"),
             self.dependencies[self.tested_reference_str].package_folder)
        save(self, os.path.join(self.build_folder, "license"),
             self.dependencies[self.tested_reference_str].license)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        if can_run(self):
            cmd = os.path.join(self.cpp.build.bindir, "test_package")
            self.run(cmd, env="conanrun")

            package_folder = load(self, os.path.join(self.build_folder, "package_folder"))
            license = load(self, os.path.join(self.build_folder, "license"))
            assert os.path.isfile(os.path.join(package_folder, "licenses", "LICENSE.txt"))
            assert license == 'BSL-1.0'
