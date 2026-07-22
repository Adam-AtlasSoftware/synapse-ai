# The Qt6/QML dashboard. Compiled by the ordinary Clang toolchain (no AdaptiveCpp)
# and links the engine as a plain library through its pure-C++ headers.

find_package(Qt6 REQUIRED COMPONENTS Quick Qml QuickControls2 Charts)

qt_standard_project_setup(REQUIRES 6.5)

# QML files live in a qml/ subdirectory; NEW makes Qt auto-generate the per-directory
# qmldir instead of warning about it.
qt_policy(SET QTP0004 NEW)

qt_add_executable(synapse_dashboard
  src/main.cpp
  src/EngineBridge.cpp
  src/EngineBridge.h
  src/ModelSession.cpp
  src/ModelSession.h
  src/AutoTrainer.cpp
  src/AutoTrainer.h
)

# Theme is a QML singleton (pragma Singleton); it must be declared as one here too or the
# generated qmldir won't match and the import fails.
set_source_files_properties(qml/Theme.qml PROPERTIES QT_QML_SINGLETON_TYPE TRUE)

qt_add_qml_module(synapse_dashboard
  URI SynapseDashboard
  VERSION 1.0
  QML_FILES
    # design system
    qml/Theme.qml
    qml/Card.qml
    qml/PrimaryButton.qml
    qml/SecondaryButton.qml
    qml/Disclosure.qml
    qml/CoachLine.qml
    qml/InfoTip.qml
    # app
    qml/Main.qml
    qml/WorkflowPanel.qml
    qml/NetworkView.qml
    qml/PixelGrid.qml
    qml/SegmentDisplay.qml
    qml/InputWidget.qml
    qml/LossChart.qml
    qml/DataManager.qml
    qml/StatsPanel.qml
    qml/CodeLab.qml
)

# Brand artwork compiled into the binary, so the app icon and in-app mark are
# available at runtime on every platform regardless of the working directory.
# Aliases (PREFIX "/" + BASE res) give clean paths like :/png/app-icon/icon-256.png
# and :/branding/mark-mono-white-1024.png.
qt_add_resources(synapse_dashboard "branding"
  PREFIX "/"
  BASE "${CMAKE_SOURCE_DIR}/res"
  FILES
    "${CMAKE_SOURCE_DIR}/res/png/app-icon/icon-16.png"
    "${CMAKE_SOURCE_DIR}/res/png/app-icon/icon-32.png"
    "${CMAKE_SOURCE_DIR}/res/png/app-icon/icon-48.png"
    "${CMAKE_SOURCE_DIR}/res/png/app-icon/icon-64.png"
    "${CMAKE_SOURCE_DIR}/res/png/app-icon/icon-128.png"
    "${CMAKE_SOURCE_DIR}/res/png/app-icon/icon-256.png"
    "${CMAKE_SOURCE_DIR}/res/png/app-icon/icon-512.png"
    "${CMAKE_SOURCE_DIR}/res/branding/mark-color-1024.png"
    "${CMAKE_SOURCE_DIR}/res/branding/mark-mono-white-1024.png"
)

# Lets the bridge find the bundled model specs at runtime during development, and
# know its own source + build directories so the Tier-2 "Code Lab" can edit engine
# C++ and rebuild. Each build (my build/, the user's out/) bakes in its own dirs.
target_compile_definitions(synapse_dashboard PRIVATE
  SYNAPSE_MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
  SYNAPSE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  SYNAPSE_BUILD_DIR="${CMAKE_BINARY_DIR}"
  # Where the out-of-process engine lives, for SYNAPSE_ENGINE_PROCESS=1.
  SYNAPSE_HOST_EXE="$<TARGET_FILE:synapse_engine_host>"
)
add_dependencies(synapse_dashboard synapse_engine_host)

target_link_libraries(synapse_dashboard PRIVATE
  synapse_engine
  Qt6::Quick
  Qt6::Qml
  Qt6::QuickControls2
  Qt6::Charts
)
