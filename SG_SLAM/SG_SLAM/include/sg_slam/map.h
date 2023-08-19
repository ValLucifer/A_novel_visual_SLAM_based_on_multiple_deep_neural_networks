//
//  Created by Lucifer on 2022/7/20.
//

#ifndef _SG_SLAM_MAP_H_
#define _SG_SLAM_MAP_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#include <memory>  // 智能指针
#include <unordered_map>  
#include <mutex>  // C++14多线程中的<mutex>提供了多种互斥操作，可以显式避免数据竞争。

#include "sg_slam/mappoint.h"
#include "sg_slam/keyframe.h"
#include "sg_slam/frame.h"

namespace sg_slam {

/**
 * @brief 地图
 * 和地图的交互： 前端调用 InsertKeyframe和InsertMapPoint插入新帧和地图点， 后端和回环维护地图的结构，判定outlier/剔除等等
 */
class Map 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h
    using this_type = Map;  // 本类别名
    using Ptr = std::shared_ptr<Map>; // 本类共享/弱指针别名
    using Landmarks_mapType = std::unordered_map<unsigned long, MapPoint::Ptr>;  // 地图点映射容器类型，key为地图点ID，value为地图点共享指针. 
    using Keyframes_mapType = std::unordered_map<unsigned long, KeyFrame::Ptr>;  // 类上
    using FrameResults_mapType = std::unordered_map<unsigned long, FrameResult::Ptr>;  // 类上

public:
    Map();
    ~Map() = default;

    // 增加一个关键帧
    void InsertKeyFrame(KeyFrame::Ptr key_frame);

    // 增加一个地图点(同时加在全局，和激活)
    void InsertMapPoint(MapPoint::Ptr map_point);

    // 增加一个激活地图点(需要将历史点重新加入激活区) // My add
    void InsertActiveMapPoint(MapPoint::Ptr map_point);

    // 增加一个 frame result
    void InsertFrameResult(FrameResult::Ptr frame_result);

    // 获取所有地图点
    Landmarks_mapType GetAllMapPoints();

    // 获取所有关键帧
    Keyframes_mapType GetAllKeyFrames();

    // 获取激活地图点
    Landmarks_mapType GetActiveMapPoints();

    // 获取激活关键帧
    Keyframes_mapType GetActiveKeyFrames();

    // 获取所有帧估计数据
    FrameResults_mapType GetAllFrameResults();

    // My Add
    // 获取 地图的当前关键帧(后端,回环使用)
    KeyFrame::Ptr GetCurrentKeyFrame();

    // 清理map中active_landmarks_map_中激活观测数量为零的点.
    void CleanActiveMap();  // 原本十四讲中的CleanMap()

    // 清理map中全局观测数量为零的点, 或激活观测为0的点
    void CleanMap();  

    // 将一个地图点从地图中删去，在回环中发生地图点融合时会使用到.
    void RemoveMapPoint(MapPoint::Ptr map_point);

private:
    // 将旧的关键帧置为不活跃状态(会调用CleanActiveMap(), 可以将激活区的地图点设为不活跃)
    void RemoveOldKeyframe();

    std::mutex data_mutex_;   // 所有映射集合的锁(除了可不加锁的)
    // std::mutex active_mutex_; 
    Landmarks_mapType landmarks_map_;  // all landmarks 
    Landmarks_mapType active_landmarks_map_;  // active landmarks // 与上不会有循环引用，所有定义为std::shared_ptr
    Keyframes_mapType keyframes_map_;  // all key-frames  
    Keyframes_mapType active_keyframes_map_;  // active key-frames // 与上不会有循环引用，所有定义为std::shared_ptr

    // N.B.: 仅前端会对其进行操作，可不加锁 (若发生段错误再加锁)
    FrameResults_mapType frame_results_map_;  // all frame results (用于保存估计的结果，方便测评)

    // N.B.: 不会同时被多线程操作，可不加锁 (若发生段错误再加锁)
    KeyFrame::Ptr current_keyframe_ = nullptr;  // 不会有循环引用，所有定义为std::shared_ptr.
    // Frame::Ptr current_frame_ = nullptr;  // N.B.: 仅用于显示.  

    // settings
    unsigned int num_active_keyframes_ = 7;     // 激活的关键帧数量(可调) // 在构造函数 Map()中设置.
    double active_keyframe_pose_min_dis_threshold_ = 0.2;  // 关键帧距离最近阈值(使用Tc1w*Twc2的李代数范数度量)(可调)

};  // class Map

}  // namespace sg_slam

#endif  // _SG_SLAM_MAP_H_
