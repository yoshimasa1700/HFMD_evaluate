# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/2.8.10.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/2.8.10.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/local/Cellar/cmake/2.8.10.2/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/masahiko/Programs/HFMD_evaluate

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/masahiko/Programs/HFMD_evaluate

# Include any dependencies generated for this target.
include CMakeFiles/objectPoseEstimation.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/objectPoseEstimation.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/objectPoseEstimation.dir/flags.make

CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o: CMakeFiles/objectPoseEstimation.dir/flags.make
CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o: objectPoseEstimation.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/objectPoseEstimation.cpp

CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/objectPoseEstimation.cpp > CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.i

CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/objectPoseEstimation.cpp -o CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.s

CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o.requires:
.PHONY : CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o.requires

CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o.provides: CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o.requires
	$(MAKE) -f CMakeFiles/objectPoseEstimation.dir/build.make CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o.provides.build
.PHONY : CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o.provides

CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o.provides.build: CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o

CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o: CMakeFiles/objectPoseEstimation.dir/flags.make
CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o: CRForest.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/CRForest.cpp

CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/CRForest.cpp > CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.i

CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/CRForest.cpp -o CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.s

CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o.requires:
.PHONY : CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o.requires

CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o.provides: CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o.requires
	$(MAKE) -f CMakeFiles/objectPoseEstimation.dir/build.make CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o.provides.build
.PHONY : CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o.provides

CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o.provides.build: CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o

CMakeFiles/objectPoseEstimation.dir/util.cpp.o: CMakeFiles/objectPoseEstimation.dir/flags.make
CMakeFiles/objectPoseEstimation.dir/util.cpp.o: util.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/objectPoseEstimation.dir/util.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/objectPoseEstimation.dir/util.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/util.cpp

CMakeFiles/objectPoseEstimation.dir/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/objectPoseEstimation.dir/util.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/util.cpp > CMakeFiles/objectPoseEstimation.dir/util.cpp.i

CMakeFiles/objectPoseEstimation.dir/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/objectPoseEstimation.dir/util.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/util.cpp -o CMakeFiles/objectPoseEstimation.dir/util.cpp.s

CMakeFiles/objectPoseEstimation.dir/util.cpp.o.requires:
.PHONY : CMakeFiles/objectPoseEstimation.dir/util.cpp.o.requires

CMakeFiles/objectPoseEstimation.dir/util.cpp.o.provides: CMakeFiles/objectPoseEstimation.dir/util.cpp.o.requires
	$(MAKE) -f CMakeFiles/objectPoseEstimation.dir/build.make CMakeFiles/objectPoseEstimation.dir/util.cpp.o.provides.build
.PHONY : CMakeFiles/objectPoseEstimation.dir/util.cpp.o.provides

CMakeFiles/objectPoseEstimation.dir/util.cpp.o.provides.build: CMakeFiles/objectPoseEstimation.dir/util.cpp.o

CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o: CMakeFiles/objectPoseEstimation.dir/flags.make
CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o: CRTree.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/CRTree.cpp

CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/CRTree.cpp > CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.i

CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/CRTree.cpp -o CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.s

CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o.requires:
.PHONY : CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o.requires

CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o.provides: CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o.requires
	$(MAKE) -f CMakeFiles/objectPoseEstimation.dir/build.make CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o.provides.build
.PHONY : CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o.provides

CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o.provides.build: CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o

CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o: CMakeFiles/objectPoseEstimation.dir/flags.make
CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o: HoG.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/HoG.cpp

CMakeFiles/objectPoseEstimation.dir/HoG.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/objectPoseEstimation.dir/HoG.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/HoG.cpp > CMakeFiles/objectPoseEstimation.dir/HoG.cpp.i

CMakeFiles/objectPoseEstimation.dir/HoG.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/objectPoseEstimation.dir/HoG.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/HoG.cpp -o CMakeFiles/objectPoseEstimation.dir/HoG.cpp.s

CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o.requires:
.PHONY : CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o.requires

CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o.provides: CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o.requires
	$(MAKE) -f CMakeFiles/objectPoseEstimation.dir/build.make CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o.provides.build
.PHONY : CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o.provides

CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o.provides.build: CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o

CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o: CMakeFiles/objectPoseEstimation.dir/flags.make
CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o: detectionresult.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/detectionresult.cpp

CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/detectionresult.cpp > CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.i

CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/detectionresult.cpp -o CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.s

CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o.requires:
.PHONY : CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o.requires

CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o.provides: CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o.requires
	$(MAKE) -f CMakeFiles/objectPoseEstimation.dir/build.make CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o.provides.build
.PHONY : CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o.provides

CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o.provides.build: CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o

# Object files for target objectPoseEstimation
objectPoseEstimation_OBJECTS = \
"CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o" \
"CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o" \
"CMakeFiles/objectPoseEstimation.dir/util.cpp.o" \
"CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o" \
"CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o" \
"CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o"

# External object files for target objectPoseEstimation
objectPoseEstimation_EXTERNAL_OBJECTS =

objectPoseEstimation-1.0.0: CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o
objectPoseEstimation-1.0.0: CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o
objectPoseEstimation-1.0.0: CMakeFiles/objectPoseEstimation.dir/util.cpp.o
objectPoseEstimation-1.0.0: CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o
objectPoseEstimation-1.0.0: CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o
objectPoseEstimation-1.0.0: CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o
objectPoseEstimation-1.0.0: CMakeFiles/objectPoseEstimation.dir/build.make
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_calib3d.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_contrib.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_core.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_features2d.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_flann.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_gpu.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_highgui.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_imgproc.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_legacy.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_ml.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_nonfree.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_objdetect.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_ocl.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_photo.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_stitching.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_superres.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_ts.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_video.dylib
objectPoseEstimation-1.0.0: /usr/local/lib/libopencv_videostab.dylib
objectPoseEstimation-1.0.0: CMakeFiles/objectPoseEstimation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable objectPoseEstimation"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/objectPoseEstimation.dir/link.txt --verbose=$(VERBOSE)
	$(CMAKE_COMMAND) -E cmake_symlink_executable objectPoseEstimation-1.0.0 objectPoseEstimation

objectPoseEstimation: objectPoseEstimation-1.0.0

# Rule to build all files generated by this target.
CMakeFiles/objectPoseEstimation.dir/build: objectPoseEstimation
.PHONY : CMakeFiles/objectPoseEstimation.dir/build

CMakeFiles/objectPoseEstimation.dir/requires: CMakeFiles/objectPoseEstimation.dir/objectPoseEstimation.cpp.o.requires
CMakeFiles/objectPoseEstimation.dir/requires: CMakeFiles/objectPoseEstimation.dir/CRForest.cpp.o.requires
CMakeFiles/objectPoseEstimation.dir/requires: CMakeFiles/objectPoseEstimation.dir/util.cpp.o.requires
CMakeFiles/objectPoseEstimation.dir/requires: CMakeFiles/objectPoseEstimation.dir/CRTree.cpp.o.requires
CMakeFiles/objectPoseEstimation.dir/requires: CMakeFiles/objectPoseEstimation.dir/HoG.cpp.o.requires
CMakeFiles/objectPoseEstimation.dir/requires: CMakeFiles/objectPoseEstimation.dir/detectionresult.cpp.o.requires
.PHONY : CMakeFiles/objectPoseEstimation.dir/requires

CMakeFiles/objectPoseEstimation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/objectPoseEstimation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/objectPoseEstimation.dir/clean

CMakeFiles/objectPoseEstimation.dir/depend:
	cd /Users/masahiko/Programs/HFMD_evaluate && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/masahiko/Programs/HFMD_evaluate /Users/masahiko/Programs/HFMD_evaluate /Users/masahiko/Programs/HFMD_evaluate /Users/masahiko/Programs/HFMD_evaluate /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles/objectPoseEstimation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/objectPoseEstimation.dir/depend

