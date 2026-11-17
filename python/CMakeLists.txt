find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(CCD REQUIRED IMPORTED_TARGET ccd)
option(COMOTION_PYTHON_ROBOT_SCENES "Build URDF and generated-robot collision bindings" ON)
set(_vamp "${PROJECT_SOURCE_DIR}/external/como-ompl/external/vamp")
if(NOT EXISTS "${_vamp}/src/impl/vamp/vector.hh")
  message(FATAL_ERROR "Initialize submodules: git submodule update --init --recursive")
endif()
if(NOT TARGET vamp_cpp)
  set(VAMP_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  set(VAMP_INSTALL_CPP_LIBRARY OFF CACHE BOOL "" FORCE)
  set(VAMP_LTO OFF CACHE BOOL "" FORCE)
  add_subdirectory("${_vamp}" "${CMAKE_CURRENT_BINARY_DIR}/vamp")
endif()
if(NOT VAMP_ARCH)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(VAMP_ARCH "-march=native -mavx2")
  else()
    set(VAMP_ARCH "-mcpu=native")
  endif()
endif()
separate_arguments(_arch_flags NATIVE_COMMAND "${VAMP_ARCH}")
function(comotion_python_module target)
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(${target} PRIVATE vamp_cpp Eigen3::Eigen)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE ${_arch_flags} -Wno-ignored-attributes)
  install(TARGETS ${target} LIBRARY DESTINATION comotion)
endfunction()
pybind11_add_module(_primitive NO_EXTRAS
  src/primitive_scene.cpp src/primitive_geometry.cpp src/primitive_queries.cpp)
# Keep the collision kernels in one translation unit for cross-function inlining.
set_target_properties(_primitive PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 0)
comotion_python_module(_primitive)
target_link_libraries(_primitive PRIVATE PkgConfig::CCD)
if(COMOTION_PYTHON_ROBOT_SCENES)
  pkg_check_modules(FCL REQUIRED IMPORTED_TARGET fcl)
  set(_src "${PROJECT_SOURCE_DIR}/src/comotion")
  pybind11_add_module(_collision NO_EXTRAS src/bindings.cpp
    "${_src}/robot/RobotModel.cpp"
    "${_src}/collision/CollisionChecker.cpp"
    "${_src}/collision/SphereCollisionBackend.cpp"
    "${_src}/collision/FclCollisionBackend.cpp"
    "${_src}/collision/VampCollisionBackend.cpp")
  comotion_python_module(_collision)
  target_link_libraries(_collision PRIVATE PkgConfig::FCL)
  target_compile_definitions(_collision PRIVATE COMOTION_HAVE_VAMP=1
    EIGEN_MAX_ALIGN_BYTES=16 EIGEN_MAX_STATIC_ALIGN_BYTES=16)
endif()
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
        DESTINATION comotion/licenses)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/LICENSES/" DESTINATION comotion/licenses)

# Read resources from the same files as the upstream C++ examples.
set(_tasks "${CMAKE_CURRENT_BINARY_DIR}/tasks")
execute_process(COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/build_resources.py"
  "${PROJECT_SOURCE_DIR}" "${_tasks}" RESULT_VARIABLE _task_status)
if(NOT _task_status EQUAL 0)
  message(FATAL_ERROR "Could not extract CoMotion task resources")
endif()
install(DIRECTORY "${_tasks}/" DESTINATION comotion/resources/tasks)
install(FILES "${PROJECT_SOURCE_DIR}/resources/panda/panda_spherized.urdf"
              "${PROJECT_SOURCE_DIR}/resources/panda/panda.srdf"
        DESTINATION comotion/resources/panda)
