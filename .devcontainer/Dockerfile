FROM colmap_devel:latest
LABEL maintainer="Andre Dueding"
ARG PYTHON_VERSION=3.11
# Combine apt-get updates and installs to reduce layers and build time
# REMOVED: libcgal-dev (included in qt5 version)
# REMOVED: libpng/jpeg/tiff (usually pulled by opencv, but kept here for safety if you need specific headers)
ENV EIGEN_BUILD_ARG="-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/"
ENV OPENMVS_BUILD_ARG="-DOpenMVS_USE_CUDA=ON -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/ -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include/ -DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64 -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/ -DCMAKE_CUDA_ARCHITECTURES=native -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3"
ENV PIP_ROOT_USER_ACTION=ignore

# Check if nvcc is available
RUN command -v nvcc || echo "nvcc not found"

RUN apt-get update -y && \
    apt-get install -y \
        unzip \
        wget \
        software-properties-common \
        git \
        build-essential \
        cmake \
        # ==================== OPENMVS DEPENDENCIES ==================== \
        libpng-dev \ 
        libjpeg-dev \
        libtiff-dev \
        libglu1-mesa-dev \
        libglew-dev \
        libglfw3-dev 
# Eigen 3.4 is standard in many distros now, but we want 3.2 for compatibility with other libs
RUN git clone https://gitlab.com/libeigen/eigen --branch 3.4 
RUN mkdir eigen_build
RUN cd eigen_build &&\
    cmake . ../eigen $EIGEN_BUILD_ARG &&\
    make && make install &&\
    cd .. && rm -rf eigen_build eigen

RUN apt-get install -y \
        # BOOST (Re-installing these is CORRECT as base image lacks headers)
        libboost-iostreams-dev \ 
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-serialization-dev \
        # OpenCV
        libopencv-dev \
        # FLANN
        libnanoflann-dev \
        # CGAL (libcgal-qt5-dev pulls in libcgal-dev)
        # libcgal-qt5-dev &&\
        # Eigen
        # libeigen3-dev && \
        # CGAL Deps
        libgmp-dev \
        libmpfr-dev \
        zlib1g-dev &&\
    update-ca-certificates
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev

# Configure git for better SSL handling
RUN git config --global http.sslVerify true && \
    git config --global http.postBuffer 524288000


RUN wget https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY requirements.txt /app/requirements.txt

# Install PyTorch
# RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu129 -r requirements.txt --no-build-isolation
RUN apt-get update &&\ 
    python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN python - <<'EOF'
import torch
print("Torch CUDA version:", torch.version.cuda)
print("Torch CUDA available:", torch.cuda.is_available())
EOF

# Install PyColmap
# RUN python -m pip install pycolmap

# From Wheel
RUN cd /pycolmap && \
    python -m pip install --no-deps --force-reinstall pycolmap_cuda12-*.whl

# 3. (Optional) Delete the wheel to save space
RUN cd /pycolmap && rm pycolmap_cuda12-*.whl

# Extra tools 
RUN python -m pip install notebook nbstripout

RUN git clone https://github.com/Driva96/CMVS-PMVS.git && \
    cd CMVS-PMVS &&\ 
    cd program && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -S .. -B . && \
    make install 
    # && \
    # cp bin/* /usr/local/bin/


# =================== OPENMVS INSTALLATION ==================== #
# =================== DEACTIVATED FOR NOW ==================== #
# # VCGLib
RUN git clone https://github.com/cdcseacave/VCG.git /vcglib

# # nanoflann
# RUN git clone --recurse-submodules https://github.com/jlblancoc/nanoflann.git /nanoflann && \
#     cd /nanoflann && \
#     mkdir build && cd build && \
#     cmake -S .. -B . && \
#     make && \
#     make install

RUN apt-get update && apt-get install -y \
    libjxl-tools \
    libjxl-devtools \
    libjxl-dev

# RUN python - <<'EOF'
# import torch
# print("Torch CUDA version:", torch.version.cuda)
# print("Torch CUDA available:", torch.cuda.is_available())
# EOF

RUN apt-get remove -y libcgal-dev && \
    apt-get autoremove -y


RUN git clone https://github.com/cgal/cgal --branch=v6.0.1 && \
    mkdir cgal_build && \
    cd cgal_build &&\
    cmake . ../cgal \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/cgal6 &&\
    make && make install &&\
    cd .. && rm -rf cgal_build cgal
# maybe use -DCGAL_HEADER_ONLY=ON \

RUN apt-get update && apt-get install -y \
    qt6-base-dev qt6-tools-dev libqt6opengl6-dev

RUN git clone https://github.com/leethomason/tinyxml2.git /tinyxml2 && \
    cd /tinyxml2 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -Dtinyxml2_SHARED_LIBS=ON && \
    cmake --build . -j$(nproc) && \
    cmake --install . && \
    rm -rf /tinyxml2

# Install TinyEXIF (manual build + install to /usr/local)
RUN git clone --depth 1 https://github.com/cdcseacave/TinyEXIF.git /tinyexif && \
    mkdir -p /tinyexif/build && cd /tinyexif/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DTinyXML2_DIR=/usr/local/lib/cmake/TinyXML2 && \
    cmake --build . -j$(nproc) && \
    cmake --install . && \
    rm -rf /tinyexif

# add to deps list above 
RUN apt-get install -y \
    libboost-exception-dev \
    libgoogle-glog-dev  

RUN git clone https://github.com/cdcseacave/TinyNPY.git /tinynpy && \
    cd /tinynpy && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON && \
    cmake --build . -j$(nproc) && \
    cmake --install . && \
    rm -rf /tinynpy

RUN git clone --recursive https://github.com/vlarsson/PoseLib.git /poselib && \
    cd /poselib && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON && \
    cmake --build . -j$(nproc) && \
    cmake --install . && \
    rm -rf /poselib

RUN git clone --recurse-submodules https://ceres-solver.googlesource.com/ceres-solver && \
    mkdir ceres-build && cd ceres-build && \
    cmake ../ceres-solver -DEIGEN_INCLUDE_DIR=/usr/local/include/eigen3 -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release  && \
    make -j$(nproc) && make install

# === OpenMVS === #
RUN git clone https://github.com/cdcseacave/openMVS.git /openMVS --branch=develop && \
    cd /openMVS && \
    git fetch https://github.com/cdcseacave/openMVS.git 973a37b2e9f417a80696a1d93994e636078587aa &&\
    git checkout FETCH_HEAD &&\
    git submodule update --init --recursive && \
    # ENABLED: Compatibility fixes
    sed -i 's/IMWRITE_JPEGXL_QUALITY/IMWRITE_JPEG_QUALITY/g' /openMVS/libs/Common/Types.inl && \
    # DISABLED: We have the correct CGAL 5 now, so no need to rename traits
    #sed -i 's/AABB_traits_3.h/AABB_traits.h/g' /openMVS/libs/MVS/SceneReconstruct.cpp && \
    #sed -i 's/AABB_triangle_primitive_3.h/AABB_triangle_primitive.h/g' /openMVS/libs/MVS/SceneReconstruct.cpp && \
    rm -rf openMVS_build && mkdir openMVS_build && \
    cd openMVS_build && \
    cmake -S .. -B . \
        -DOpenMVS_MAX_CUDA_COMPATIBILITY=ON \
        -DBLA_VENDOR=Intel10_64lp \
        -DCMAKE_CUDA_ARCHITECTURES=61 \
        -DOpenMVS_USE_QT=ON \
        -DQt6_DIR=/usr/lib/x86_64-linux-gnu/cmake/Qt6 \
        -DCMAKE_BUILD_TYPE=Release \
        -DVCG_ROOT=/vcglib \
        $OPENMVS_BUILD_ARG \
        -DCGAL_DIR=/opt/cgal6/lib/cmake/CGAL \
        -DTinyEXIF_DIR=/usr/local/lib/cmake/TinyEXIF && \
    cmake --build . -j4 --config Release && \
    cmake --install . && \
    cd .. && rm -rf openMVS_build /vcglib

# === MVS-Texturing ===
# Deps 
RUN apt-get update && apt-get install -y \
    libtbb-dev 

 RUN git clone https://github.com/Driva96/mvs-texturing.git /mvs-texturing && \
    sed -i 's/unsigned int padding = size >> 7;/unsigned int padding = size >> 8;/g' /mvs-texturing/libs/tex/generate_texture_atlases.cpp &&\
    cd /mvs-texturing && \
    mkdir build && cd build && \
    cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release && \
    make -j4 && \
    make install && \
    cd .. && rm -rf build


# git fetch https://github.com/cdcseacave/openMVS.git "v2.3.0" && \
# git checkout FETCH_HEAD && \

# Need to move further up in final image
# RUN python -m pip install scikit-image

COPY . /app