# openfabmap-python
Boost Python bindings for openFABMAP, see [https://github.com/arrenglover/openfabmap](https://github.com/arrenglover/openfabmap)

Requires:
- OpenFABMAP, download from [https://github.com/arrenglover/openfabmap](https://github.com/arrenglover/openfabmap)
- OpenCV 2.3 or higher, with nonfree additions
- Boost Python

Building:
  cd /path/to/source
  mkdir build
  cd build
  cmake -DOpenCV_DIR=/path/to/opencv -DPY_INSTALL_DIR=/python/install/path ..
  make
  make install

