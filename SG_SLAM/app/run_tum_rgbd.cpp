//
//  Created by Lucifer on 2022/10/13.
//

#include <gflags/gflags.h>

#include <iostream>

#include "sg_slam/system.h"

DEFINE_string( config_file, "./config/TUM_default.yaml", "config file path.");

int main(int argc, char **argv) 
{
    std::cout << "argc: " << argc << std::endl;
    google::SetUsageMessage("usage:");
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << "argc: " << argc << std::endl;
    std::cout << "config_file: " << FLAGS_config_file << std::endl;

    sg_slam::System::Ptr slam(new sg_slam::System( FLAGS_config_file ));

    assert(slam->Init() == true);

    slam->Run();

    slam->Stop();

    // save the keyframe trajectory
    std::string saveFile1 = "./result/TUM/Keyframe_trajectory_TUM.txt";
    slam->SaveKeyframeTrajectoryTUM(saveFile1);

    std::string saveFile2 = "./result/TUM/loop.txt";
    slam->SaveLoopEdges(saveFile2);

    std::string saveFile3 = "./result/TUM/frame_trajectory_TUM.txt";
    slam->SaveFrameTrajectoryTUM(saveFile3);

    std::string saveMappointPCD_file = "./result/TUM/MappointPCD.pcd";
    slam->SaveMappointPCD(saveMappointPCD_file);

    std::cout << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << " main finish! " << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;

    return 0;
}
