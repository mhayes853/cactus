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
    echo "Building Cactus for all Apple platforms (ARM only)..."
    echo "Build type: $CMAKE_BUILD_TYPE"
    echo "Using $n_cpu CPU cores"

    APPLE_OUT="$SCRIPT_DIR/build"
    XCFRAMEWORK_PATH="$SCRIPT_DIR/CXXCactusDarwin.xcframework"

    rm -rf "$APPLE_OUT" "$XCFRAMEWORK_PATH"
    mkdir -p "$APPLE_OUT"

    function build_apple_target() {
        local PLATFORM=$1        # iOS / iOS_SIM / macOS / tvOS / watchOS / visionOS
        local SYS=$2             # CMake system (iOS / watchOS / tvOS / visionOS / Darwin)
        local SDK=$3             # SDK identifier (iphoneos / iphonesimulator / etc)
        local ARCH="arm64"
        local OUT="$APPLE_OUT/$PLATFORM"
        local VERSION=$4

        local SDK_PATH
        SDK_PATH=$(xcrun --sdk "$SDK" --show-sdk-path)

        echo "▶️  Building $PLATFORM ($SYS, $ARCH, $SDK)"

        cmake -S "$SCRIPT_DIR" \
            -B "$OUT" \
            -GXcode \
            -DCMAKE_SYSTEM_NAME="$SYS" \
            -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
            -DCMAKE_OSX_SYSROOT="$SDK_PATH" \
            -DCMAKE_OSX_DEPLOYMENT_TARGET=$VERSION \
            -DBUILD_SHARED_LIBS=ON \
            -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DCMAKE_IOS_INSTALL_COMBINED=YES

        cmake --build "$OUT" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu"
    }

    function find_framework() {
        local OUT="$APPLE_OUT/$1"
        local FRAMEWORK=$(find "$OUT" -type d -name "CXXCactusDarwin.framework" | head -n 1)
        if [ ! -d "$FRAMEWORK" ]; then
            echo "❌ Failed: $1 build did not produce framework"
            exit 1
        fi
        echo $FRAMEWORK
    }

    echo "🛠️ Building iOS"
    build_apple_target "ios" "iOS" "iphoneos" 13.0
    IOS=$(find_framework "ios")

    echo "🛠️ Building iOS Simulator"
    build_apple_target "ios_sim" "iOS" "iphonesimulator" 13.0
    IOS_SIM=$(find_framework "ios_sim")

    echo "🛠️ Building macOS"
    build_apple_target "macos" "Darwin" "macosx" 11.0
    MAC=$(find_framework "macos")

    echo "🛠️ Building tvOS"
    build_apple_target "tvos" "tvOS" "appletvos" 13.0
    TVOS=$(find_framework "tvos")

    echo "🛠️ Building tvOS Simulator"
    build_apple_target "tvos_sim" "tvOS" "appletvsimulator" 13.0
    TVOS_SIM=$(find_framework "tvos_sim")

    echo "🛠️ Building watchOS"
    build_apple_target "watchos" "watchOS" "watchos" 6.0
    WATCHOS=$(find_framework "watchos")

    echo "🛠️ Building watchOS Simulator"
    build_apple_target "watchos_sim" "watchOS" "watchsimulator" 6.0
    WATCHOS_SIM=$(find_framework "watchos_sim")


    # VISIONOS=$(build_apple_target "visionos" "visionOS" "xros" 1.0)
    # VISIONOS_SIM=$(build_apple_target "visionos_sim" "visionOS" "xrsimulator" 1.0)

    echo "IOS: $IOS"
    echo "IOS_SIM: $IOS_SIM"
    echo "MAC: $MAC"
    echo "TVOS: $TVOS"
    echo "TVOS_SIM: $TVOS_SIM"
    echo "WATCHOS: $WATCHOS"
    echo "WATCHOS_SIM: $WATCHOS_SIM"

    echo "📦 Creating XCFramework..."

    xcodebuild -create-xcframework \
        -framework "$IOS" \
        -framework "$IOS_SIM" \
        -framework "$MAC" \
        -framework "$TVOS" \
        -framework "$WATCHOS" \
        -framework "$WATCHOS_SIM" \
        -output "$XCFRAMEWORK_PATH"

    echo "✅ Apple XCFramework built:"
    echo "   $XCFRAMEWORK_PATH"
}

build_android_artifactbundle
build_apple_xcframework

rm "$SCRIPT_DIR/cactus.h"
