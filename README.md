### Library compilation commands

#### HIP/AMD

``````
hipcc -O3 -shared -fPIC -std=c++17 systems.cpp -o systems.so
``````

CUDA/NVIDIA

``````
nvcc -shared -O3 -Xcompiler -fPIC -o systems.so systems.cu
``````

