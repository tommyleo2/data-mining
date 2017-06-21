project(GBDT)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_BUILD_TYPE Debug)

find_package(GTest)
find_package(Threads REQUIRED)

include(CTest)

include_directories("include")
file(GLOB FRAMEWORK_SRC_FILES "${PROJECT_SOURCE_DIR}/src/framework/*.cpp")
file(GLOB IMPL_SRC_FILES "${PROJECT_SOURCE_DIR}/src/in-memory-impl/*.cpp")
file(GLOB BOOT_FILE "${PROJECT_SOURCE_DIR}/src/main.cpp")

if (BUILD_TESTING)

  file(GLOB TEST_SRC_FILES "${PROJECT_SOURCE_DIR}/test/*.cpp")
  #  Add test executable
  add_executable(${PROJECT_NAME}Test ${TEST_SRC_FILES} ${FRAMEWORK_SRC_FILES} ${IMPL_SRC_FILES})
  set_target_properties(${PROJECT_NAME}Test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    CXX_STANDARD 14
    )

  target_link_libraries(${PROJECT_NAME}Test
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    )

  add_test(test ${PROJECT_NAME}Test)

endif ()

add_executable(${PROJECT_NAME} ${FRAMEWORK_SRC_FILES} ${IMPL_SRC_FILES} ${BOOT_FILE})
set_target_properties(${PROJECT_NAME}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
  CXX_STANDARD 14
  )

target_link_libraries(${PROJECT_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
  )
