# ZED SDK container for running zed2dataset.py without a native SDK install
# (Stereolabs targets Ubuntu; this keeps the SVO->dataset pipeline off the
# host OS entirely). The stereolabs/zed image already ships Python 3.12 +
# pyzed + the SDK; we only add the converter's extra Python deps.
#
# numpy is PINNED to the image's version: pyzed is a compiled extension
# built against that numpy, and letting pip upgrade it breaks the ABI.
FROM stereolabs/zed:5.4-runtime-cuda13.0-ubuntu24.04

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "numpy==2.4.6" opencv-python-headless pandas

WORKDIR /workspace
