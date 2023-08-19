add_executable( run_kitti_stereo run_kitti_stereo.cpp )
target_link_libraries( run_kitti_stereo sgslam SuperShared LoopCloseShared ${THIRD_PARTY_LIBS})

add_executable( run_tum_rgbd run_tum_rgbd.cpp )
target_link_libraries( run_tum_rgbd sgslam SuperShared LoopCloseShared ${THIRD_PARTY_LIBS})

add_executable( run_EuRoC_stereo run_EuRoC_stereo.cpp )
target_link_libraries( run_EuRoC_stereo sgslam SuperShared LoopCloseShared ${THIRD_PARTY_LIBS})
