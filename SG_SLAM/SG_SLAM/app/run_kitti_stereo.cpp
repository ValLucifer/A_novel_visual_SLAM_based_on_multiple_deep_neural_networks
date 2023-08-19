//
//  Created by Lucifer on 2022/8/13.
//

#include <gflags/gflags.h>

#include <iostream>

#include "sg_slam/system.h"

#include "sg_slam/viewer.h"  // 20221028

DEFINE_string( config_file, "./config/default.yaml", "config file path.");

DEFINE_string( frame_trajectory, "./result/frame_trajectory.txt", "output file path." );

int main(int argc, char **argv) 
{
    std::cout << "argc: " << argc << std::endl;
    google::SetUsageMessage("usage:");
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << "argc: " << argc << std::endl;
    std::cout << "config_file: " << FLAGS_config_file << std::endl;
    std::cout << "frame_trajectory: " << FLAGS_frame_trajectory << std::endl;

    sg_slam::Viewer::Viewer_sz = 1.0f;  // 20221028

    sg_slam::System::Ptr slam(new sg_slam::System( FLAGS_config_file ));

    slam->m_stereo_dataset_type = 0;  // 20221028 这样实现不好
    slam->m_key_max_id_interval = 20;  // 20221030 这样实现不好

    assert(slam->Init() == true);

    slam->Run();
    
    /**
    char c = 'A';
    std::cout << " Please enter q to stop: ";
    while(std::cin>>c && c!='q') {

    }
    **/

    slam->Stop();

    /**
    // save the keyframe trajectory
    std::string saveFile1 = "./result/Keyframe_trajectory.txt";
    slam->SaveKeyframeTrajectory(saveFile1);

    std::string saveFile2 = "./result/Keyframe_trajectory_TUM.txt";
    slam->SaveKeyframeTrajectoryTUM(saveFile2);

    std::string saveFile3 = "./result/loop.txt";
    slam->SaveLoopEdges(saveFile3);

    std::string saveFile4 = "./result/frame_trajectory_TUM.txt";
    slam->SaveFrameTrajectoryTUM(saveFile4);

    std::string saveFile5 = "./result/frame_trajectory_KITTI.txt";
    slam->SaveFrameTrajectoryKITTI(saveFile5);

    std::string saveFile6 = "./result/frame_trajectory_KITTI_no_y.txt";
    slam->SaveFrameTrajectoryKITTI_no_y(saveFile6);

    std::string saveMappointPCD = "./result/MappointPCD.pcd";
    slam->SaveMappointPCD(saveMappointPCD);
    **/

    // output
    slam->SaveFrameTrajectoryKITTI(FLAGS_frame_trajectory);

    std::cout << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << " main finish! " << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;

    return 0;
}
