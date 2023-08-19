//
//  Created by Lucifer on 2022/10/28.
//

#include <gflags/gflags.h>

#include <iostream>

#include "sg_slam/system.h"

#include "sg_slam/viewer.h"  // 20221028

DEFINE_string( config_file, "./config/EuRoCdefault.yaml", "config file path.");

DEFINE_string( frame_trajectory, "./result/EuRoC/frame_trajectory.txt", "output file path." );
DEFINE_string( Keyframe_trajectory, "./result/EuRoC/Keyframe_trajectory.txt", "output file path." );

int main(int argc, char **argv) 
{
    std::cout << "argc: " << argc << std::endl;
    google::SetUsageMessage("usage:");
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << "argc: " << argc << std::endl;
    std::cout << "config_file: " << FLAGS_config_file << std::endl;
    std::cout << "frame_trajectory: " << FLAGS_frame_trajectory << std::endl;
    std::cout << "Keyframe_trajectory: " << FLAGS_Keyframe_trajectory << std::endl;

    sg_slam::Viewer::Viewer_sz = 0.01f;

    sg_slam::System::Ptr slam(new sg_slam::System( FLAGS_config_file ));

    slam->m_stereo_dataset_type = 1;  // 20221028 这样实现不好
    slam->m_key_max_id_interval = 10;  // 10 // 20221030 这样实现不好
    slam->m_local_keyframes_deque_length = 2; // 2 // 20221104 这样实现不好

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
    std::string saveFile2 = "./result/EuRoC/Keyframe_trajectory_TUM.txt";
    slam->SaveKeyframeTrajectoryTUM(saveFile2);

    std::string saveFile3 = "./result/EuRoC/loop.txt";
    slam->SaveLoopEdges(saveFile3);

    std::string saveFile4 = "./result/EuRoC/frame_trajectory_TUM.txt";
    slam->SaveFrameTrajectoryTUM(saveFile4);

    std::string saveMappointPCD = "./result/EuRoC/MappointPCD.pcd";
    slam->SaveMappointPCD(saveMappointPCD);
    **/

    // output
    slam->SaveFrameTrajectoryTUM(FLAGS_frame_trajectory);
    slam->SaveKeyframeTrajectoryTUM(FLAGS_Keyframe_trajectory);
    // std::string saveFile3 = "./result/EuRoC/loop.txt";
    // slam->SaveLoopEdges(saveFile3);

    std::cout << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << " main finish! " << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;

    return 0;
}
