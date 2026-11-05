include(GoogleTest)

file(GLOB MLINALG_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(mlinalg_gtest ${MLINALG_TEST_SOURCES})
target_link_libraries(mlinalg_gtest PRIVATE mlinalg::mlinalg GTest::gtest)
target_compile_options(mlinalg_gtest PRIVATE -Wall -Wextra)
set_target_properties(mlinalg_gtest PROPERTIES CXX_EXTENSIONS OFF)

# The default 5s limit is easy to exceed on a loaded CI runner under a sanitizer.
gtest_discover_tests(mlinalg_gtest DISCOVERY_TIMEOUT 60)
