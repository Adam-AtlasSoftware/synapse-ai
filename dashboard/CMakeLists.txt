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
)

qt_add_qml_module(synapse_dashboard
  URI SynapseDashboard
  VERSION 1.0
  QML_FILES
    qml/Main.qml
    qml/NetworkView.qml
    qml/PixelGrid.qml
    qml/LossChart.qml
)

# Lets the bridge find the bundled model specs at runtime during development.
target_compile_definitions(synapse_dashboard PRIVATE
  SYNAPSE_MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
)

target_link_libraries(synapse_dashboard PRIVATE
  synapse_engine
  Qt6::Quick
  Qt6::Qml
  Qt6::QuickControls2
  Qt6::Charts
)
