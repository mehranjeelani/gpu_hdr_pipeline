
Framework for the [127081 GPU Programming][1] practical exercises

## Building

The project can run on Linux or Windows 10 given the following prerequisites

  * [CMake][2] 3.18+
  * [CUDA Toolkit][3] 10.x+ (installation guides for [Linux][4] and [Windows][5])
  * C++17 toolchain (gcc 9.x+, clang 9.x+, or Visual Studio 2019)
  * [Git][6] (with [LFS][7])
  * OpenGL 4.3 (interactive mode only)

Use CMake to generate a build system (scripts are located in `src`).
The project can be configured to build in "interactive mode" by setting the `INTERACTIVE` option to `ON`. In this configuration, running the executable for each task will open a window with an interactive display (requires OpenGL). If this option is left off, output will be written to files (this works without an OpenGL installation).

Be sure to set the GPU architecture to build for via the [`CMAKE_CUDA_ARCHITECTURES` variable][8]. For example, when targeting a GPU with compute capability `6.1`, set `CMAKE_CUDA_ARCHITECTURES` to `61`. For a list of GPUs and their compute capability, see [here][9].

### Examples

Linux
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=70 ../src
```
and then
```
make hdr_pipeline
```

Windows using Visual Studio 2019
```
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -Ax64 -DCMAKE_CUDA_ARCHITECTURES=61 -DINTERACTIVE=ON ../src
```
and then open the generated `.sln`.


## Running

The `hdr_pipeline` project for tasks 1, 2, and 3 expects the HDR image to work on as a command line argument. A number of HDR images for testing can be found in the `assets` directory. For example:
```
./build/bin/hdr_pipeline assets/LA_Downtown_Helipad_GoldenHour_3k.hdr
```
In interactive mode, HDR pipeline output will be displayed in a window in real-time. Furthermore, the list of input files may also specify `.obj` files that contain 3D geometry to be added to the scene. In noninteractive mode, HDR pipeline output will be written to `<filename>.png`.

The complete set of options for `hdr_pipeline` is
```
hdr_pipeline [{options}] {<input-file>}
	options:
	  --device <i>           use CUDA device <i>, default: 0
	  --exposure <v>         set exposure value to <v>, default: 0.0
	  --brightpass <v>       set brightpass threshold to <v>, default: 0.9
	  --test-runs <N>        average timings over <N> test runs, default: 1
```



[1]: https://graphics.cg.uni-saarland.de/courses/gpu-2020/index.html
[2]: https://cmake.org/
[3]: https://developer.nvidia.com/cuda-toolkit
[4]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
[5]: https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#windows
[6]: https://git-scm.com/
[7]: https://git-lfs.github.com/
[8]: https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
[9]: https://developer.nvidia.com/cuda-gpus#compute
