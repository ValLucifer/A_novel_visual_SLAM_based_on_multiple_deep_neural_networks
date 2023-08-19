//
//  Created by Lucifer on 2022/7/19.
//

#ifndef _SG_SLAM_KEYFRAME_H_
#define _SG_SLAM_KEYFRAME_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#include <memory>  // 参见 sg_slam/frame.h
#include <mutex>  // C++14多线程中的<mutex>提供了多种互斥操作，可以显式避免数据竞争。

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "sg_slam/common_include.h"

#include "sg_slam/deeplcd.h"

#include "sg_slam/feature.h"

namespace sg_slam {

// class Frame;
// class Feature;  // N.B.: 使用这个声明仅能使用名字, 使用包含头文件vscode才能跟踪类中的成员.

/**
 * @brief KeyFrame(关键帧)， 每一个关键帧分配独立的关键帧ID(id_)(在关键帧序列中的ID).
 * 
 */
class KeyFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h
    using this_type = KeyFrame;  // 本类别名;
    using Ptr = std::shared_ptr<KeyFrame>;  // 本类共享/弱指针别名

    unsigned long id_ = 0;  // id of this keyframe  // 非负(在关键帧序列中的ID)
    double time_stamp_;  // 时间戳, 用以将结果保存为TUM真值轨迹格式. 参见 sg_slam/frame.h (由Frame得到)
    unsigned long src_frame_id_ = 0;  // N.B: 先保存生成源帧的ID，根据后面系统运行的需要，再判断是否改为源帧的弱指针.

    // N.B.: 需多线程维护要加互斥锁.
    SE3 pose_;  // pose with form of Tcw.
    std::mutex pose_mutex_;  // Data lock of pose

    // N.B.: 只在前端赋值(使用clone()), 在回环中使用(用于生成deepLCD 回环描述向量, 及回环匹配显示)
    // N.B.: 不会同时被多个线程操作，故不用加锁，发生段错误时再加锁!!!!!
    // // N.B.: 原清华工程的 mImageLeft
    cv::Mat left_img_;

    // 2022101101 加
    bool is_use_cut_image = false;
    cv::Mat left_cut_img_;
    
    // N.B.: keyframe 中的特征点和描述子只保存左图的. (由Frame得到) 参见 sg_slam/frame.h
    std::vector< cv::KeyPoint > mvKeys;
    std::vector< cv::KeyPoint > mvKeysUn;

    // My add 20220913
    std::vector< float > mvKeysR_u;  // 与 mvKeys 大小一样，没有匹配的为-1. 非-1在初始化时与mvpfeatures非nullptr项对应 

    // Superpoint+kenc descriptor [b, d, n]/[c, h, w] (使用)
    // N.B: 参见 sg_slam/frame.h
    int64_t Descripters_B = 1;
    int64_t Descripters_D = 256;
    int64_t Descripters_N = 0;
    std::vector<float> mDescriptors_data;

    // Feature 中包含 MapPoints， Feature associated to , nullptr if no association.
    // 2D-3D, 2D in keyframe pixel coordinate(keypoints_un), 3D(MapPoint) in world coordinate.
    // 只保存左图的. 
    // N.B: 参见 sg_slam/frame.h
    // // N.B.: 原清华工程的 mvpFeaturesLeft
    // N.B.: 生成及结构改变全在前端， 后端和回环只是修改mvpfeatures中指针中的数据，不会改变这个结构故不用加锁，
    // 发生段错误可以考虑加锁
    std::vector< std::shared_ptr<Feature> > mvpfeatures;
    int keypoints_number = 0;  // N.B.: 改为Feature数更好

    // for pose graph optimization (前端赋值，回环使用)

    std::weak_ptr< KeyFrame > mpNextKF;  // 子关键帧 (下一帧)
    SE3 m_relative_pose_to_nextKF;  // N.B.: 仅回环使用可以不加锁吗?
    std::weak_ptr< KeyFrame > mpLastKF;  // 父关键帧
    // N.B.: 本关键帧相对于上一关键帧的相对位姿
    SE3 m_relative_pose_to_lastKF;  // N.B.: 仅回环使用可以不加锁吗?

    // loop closing members
    bool isLoop = false;  // N.B.: 是否有回环标志，在回环线程中设置，暂时没有，或可用于显示(My Add)
    std::weak_ptr< KeyFrame > mpLoopKF;  // 与此帧形成回环的关键帧
    // N.B.: 回环修正后的相对于回环帧的相对位姿.
    SE3 m_relative_pose_to_loopKF;  // N.B.: 回环检查及矫正使用可以不加锁，可以不用吗?

    // // pyramid keypoints only for computing ORB descriptors and doing matching.
    // std::vector< cv::KeyPoint > mvPyramidKeyPoints;
    // cv::Mat mORBDescriptors;
    DeepLCD::DescrVector mpDeepDescrVector;

public:
    KeyFrame();
    ~KeyFrame() = default;

    // set and get pose, thread safe
    SE3 Pose();
    void SetPose(const SE3 &pose);

    /// 工厂构建模式，分配id
    /// create KeyFrame and assign id of keyframe
    static KeyFrame::Ptr CreateKeyFrame();

};  // KeyFrame

}  // namespace sg_slam

#endif  // _SG_SLAM_KEYFRAME_H_
