#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
ANDROID_DIR="$ROOT_DIR/android"
SOURCE_DIR="$ROOT_DIR/cactus"

cp "$SOURCE_DIR/ffi/cactus_ffi.h" "$SCRIPT_DIR/cactus.h"

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found, please install it"
    exit 1
fi

if ! xcode-select -p &> /dev/null; then
    echo "Error: Xcode command line tools not found"
    echo "Install with: xcode-select --install"
    exit 1
fi

n_cpu=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

function build_android_artifactbundle() {
    rm -drf "$SCRIPT_DIR/CXXCactus.artifactbundle"
    echo "Building Cactus artifactbundle for Android platforms..."
    $ANDROID_DIR/build.sh

    mkdir "$SCRIPT_DIR/CXXCactus.artifactbundle"
    mkdir -p "$SCRIPT_DIR/CXXCactus.artifactbundle/dist/android"
    cp -r "$ANDROID_DIR/libcactus.a" "$SCRIPT_DIR/CXXCactus.artifactbundle/dist/android/libcactus.a"
    mkdir -p "$SCRIPT_DIR/CXXCactus.artifactbundle/include"
    cp -r "$SCRIPT_DIR/cactus.h" "$SCRIPT_DIR/CXXCactus.artifactbundle/include/cactus.h"

    cat > "$SCRIPT_DIR/CXXCactus.artifactbundle/include/module.modulemap" << 'EOF'
module CXXCactus {
    header "cactus.h"
    export *
}
EOF

    cat > "$SCRIPT_DIR/CXXCactus.artifactbundle/info.json" << 'EOF'
{
  "schemaVersion": "1.0",
  "artifacts": {
    "cxxcactus": {
      "type": "staticLibrary",
      "version": "1.0.0",
      "variants": [
        {
          "path": "dist/android/libcactus.a",
          "supportedTriples": ["aarch64-unknown-linux-android"],
          "staticLibraryMetadata": {
            "headerPaths": ["include"],
            "moduleMapPath": "include/module.modulemap"
          }
        }
      ]
    }
  }
}
EOF
}

function build_apple_xcframework() {
    echo "Building Cactus for Apple platforms..."
    echo "Build type: $CMAKE_BUILD_TYPE"
    echo "Using $n_cpu CPU cores"
}

build_android_artifactbundle
build_apple_xcframework

rm "$SCRIPT_DIR/cactus.h"
