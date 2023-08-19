//
//  Created by Lucifer on 2022/8/13.
//

#ifndef _SG_SLAM_SYSTEM_H_
#define _SG_SLAM_SYSTEM_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#include <memory>  // 智能指针
#include <string>

#include "sg_slam/KITTI_dataset.h"
#include "sg_slam/map.h"
#include "sg_slam/frontend.h"
#include "sg_slam/backend.h"
#include "sg_slam/loopclosing.h"

#include "sg_slam/camera.h"  // 20221009 加

namespace sg_slam {

/**
 * @brief SLAM 系统对外接口
 * 
 */
class System 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h
    using this_type = System;  // 本类别名
    using Ptr = std::shared_ptr<System>;  // 本类共享/弱指针别名


    // 20221009 加
    CameraType m_camera_type = CameraType::SETERO;

    // 20221028 add
    int m_stereo_dataset_type = 0;  // 0: KITTI, 1: EuRoC

    // 20221012 加
    bool m_is_use_loopclosing_flag = true;

    // 20221030 add
    unsigned long m_key_max_id_interval = 10;

    // 20221104 add
    size_t m_local_keyframes_deque_length = 2;


    /**
     * @brief constructor with config file
     * 
     */
    System(std::string &config_path);
    // ~System() = default;

    /**
     * @brief do initialization things before run
     * @return true if success
     * 
     */
    bool Init();

    /**
     * @brief start slam in the dataset
     * 
     */
    void Run();

    /**
     * @brief make a step forward in dataset (在数据集上迈出一步)
     * 
     */
    bool Step();

    /**
     * @brief stop the system (注意停止的顺序)
     * N.B.: 已停止的线程不能再被调用
     * 
     */
    void Stop();


    // N.B.: My Add 根据清华工程设计
    /**
     * @brief save the keyframe trajectory to txt file ()
     * the output format is like:  
     *      "keyframe id, timestamp, tx, ty, tz, qx, qy, qz, qw" per line
     *      (类TUM)
     */
    void SaveKeyframeTrajectory(std::string &save_file);

    /**
     * @brief save the keyframe trajectory to txt file 
     * the output format is TUM :  
     *      "timestamp, tx, ty, tz, qx, qy, qz, qw" per line
     */
    void SaveKeyframeTrajectoryTUM(std::string &save_file);

    /**
     * @brief save the edges detected by loop closure,
     *  the result of loop detection
     *  the output format is like: 
     *          "Current id, timestamp, tx, ty, tz, qx, qy, qz, qw" 
     *          "Loop id, timestamp, tx, ty, tz, qx, qy, qz, qw"
     *          two lines as a group
     */
    void SaveLoopEdges(std::string &save_file);

    // N.B.: 自加
    /**
     * @brief save the frame trajectory to txt file
     * the output format is TUM :  
     *      "timestamp, tx, ty, tz, qx, qy, qz, qw" per line
     * 
     */
    void SaveFrameTrajectoryTUM(std::string &save_file);

    // N.B.: 自加
    /**
     * @brief save the frame trajectory to txt file
     * the output format is KITTI : 
     * 
     */
    void SaveFrameTrajectoryKITTI(std::string &save_file);

    void SaveFrameTrajectoryKITTI_no_y(std::string &save_file);

    // 20221012
    /**
     * @brief save the mappoint to pcd file (no rgb, no gray)
     * 
     */
    void SaveMappointPCD(std::string &save_file);


private:
    // // 原系统 inited_
    // bool m_inited_ = false; 
    // // 原系统 config_file_path_
    std::string m_config_file_path_;

    // dataset
    // // 原系统 dataset_
    // KITTI_Dataset::Ptr m_dataset_ = nullptr;
    Dataset::Ptr m_dataset_ = nullptr;

    // // 原系统 map_
    Map::Ptr m_map_ = nullptr;
    // // 原系统 frontend_
    Frontend::Ptr m_frontend_ = nullptr;
    // // 原系统 backend_
    Backend::Ptr m_backend_ = nullptr;
    LoopClosing::Ptr m_loopclosing_ = nullptr;
    // // 原系统 viewer_
    Viewer::Ptr m_viewer_ = nullptr;


};  // class System

}  // namespace sg_slam

#endif  // _SG_SLAM_SYSTEM_H_
