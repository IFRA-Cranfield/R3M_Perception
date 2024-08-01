cmake_minimum_required(VERSION 3.5)
project(r3m_perception)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)AddTwoInts
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# ===== PYTHON ===== #
install(PROGRAMS  
  
  oneshotdetection/osd_test.py
  oneshotdetection/osd.py
  oneshotdetection/render.py

  megapose6d/m6d_test.py
  megapose6d/m6d.py

  oneshotdetection/OSD_server.py
  oneshotdetection/OSD_client.py
  megapose6d/M6D_server.py
  megapose6d/M6D_client.py

  python/ROS2_client.py
  python/perception.py

  DESTINATION lib/${PROJECT_NAME} 
)

# ===== DIRECTORIES ===== #
install(DIRECTORY
  
  cad
  config
  megapose6d
  oneshotdetection
  pictures
  urdf
  python

  DESTINATION share/${PROJECT_NAME}/
)

ament_package()