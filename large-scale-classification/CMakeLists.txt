project(GBDT)
cmake_minimum_required(VERSION 2.6)

include_directories(include)

set(CMAKE_BUILD_TYPE Debug)

find_package(GTest)
find_package(Threads REQUIRED)

enable_testing()

file(GLOB TEST_SRC_FILES "${PROJECT_SOURCE_DIR}/test/*.cpp")
file(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/framework/*.cpp")

add_executable(${PROJECT_NAME}Test ${TEST_SRC_FILES} ${SRC_FILES})

set_target_properties(${PROJECT_NAME}Test
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    CXX_STANDARD 14
)

target_compile_features(${PROJECT_NAME}Test PRIVATE cxx_range_for)
target_link_libraries(${PROJECT_NAME}Test
  ${GTEST_BOTH_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  )

add_test(test ${PROJECT_NAME}Test)
