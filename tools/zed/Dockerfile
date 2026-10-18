# ZED SDK container for tools/zed/export_svo.py. Stereolabs ships the SDK for
# Ubuntu only; this keeps SVO export off the host OS entirely. The base image
# already has Python 3.12 + pyzed + the SDK; we only add OpenCV.
#
# numpy is PINNED to the image's version: pyzed is a compiled extension built
# against it, and letting pip upgrade numpy breaks the ABI.
FROM stereolabs/zed:5.4-runtime-cuda13.0-ubuntu24.04

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "numpy==2.4.6" opencv-python-headless

WORKDIR /workspace
