add_library(SuperShared SHARED
        sg_slam/SG_superpoint.cpp
        sg_slam/superglue.cpp)
target_link_libraries(SuperShared ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_library(LoopCloseShared SHARED
        # sg_slam/ORBextractor.cpp
        sg_slam/deeplcd.cpp)
target_link_libraries(LoopCloseShared ${THIRD_PARTY_LIBS})

add_library( sgslam SHARED
        sg_slam/mappoint.cpp
        sg_slam/config.cpp
        sg_slam/camera.cpp
        sg_slam/KITTI_dataset.cpp
        sg_slam/TUM_dataset.cpp  # 2022100801
        sg_slam/EuRoC_dataset.cpp  # 2022102701
        #
        sg_slam/frame.cpp
        sg_slam/feature.cpp
        sg_slam/keyframe.cpp
        #
        sg_slam/sg_detectMatcher.cpp
        sg_slam/map.cpp
        sg_slam/viewer.cpp
        sg_slam/frontend.cpp
        sg_slam/backend.cpp
        sg_slam/loopclosing.cpp
        #
        sg_slam/system.cpp
        )

target_link_libraries(sgslam SuperShared LoopCloseShared ${THIRD_PARTY_LIBS})
