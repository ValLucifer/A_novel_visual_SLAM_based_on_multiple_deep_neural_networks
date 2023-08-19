//
//  Created by Lucifer on 2022/7/20.
//

#ifndef _SG_SLAM_MAPPOINT_H_
#define _SG_SLAM_MAPPOINT_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#include <memory>  // 智能指针
#include <mutex>  // 详见 sg_slam/frame.h
// #include <vector>  // 详见 sg_slam/frame.h
#include <list>  // 列表是C++标准库容器之一

#include "sg_slam/common_include.h"

namespace sg_slam {

// forward declare
class Feature;
class KeyFrame;

/**
 * @brief 路标点类, 特征点在三角化之后形成路标点
 * 类中数据(3D位置(pos_)和观测(observations_,observed_times_))共用一个互斥锁故不可以(多线程)同时操作.
 */
class MapPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 详见 sg_slam/frame.h
    using Ptr = std::shared_ptr<MapPoint>;

    unsigned long id_ = 0;  // ID
    std::weak_ptr<KeyFrame> reference_KeyFrame_;  // 生成(参考关键帧)指针 // N.B.: 赋值之后就不变了.
    Vec3 pos_ = Vec3::Zero(); // Position in world (pw)
    // N.B.: 设计时人为规定同一张图片上不会有多个2D点对应同一个地图点, 在优化合并地图点时要注意处理.
    // 不用考虑多次添加情况.
    // 参见 src/sg_slam/mappoint.cpp RemoveObservation.
    // N.B.: 个人认为最好将list改为unorder_map/unorder_set
    std::list< std::weak_ptr<Feature> > observations_; // shared_ptr<Feature> 在KeyFrame中. 
    // 被关键帧观测到的次数 (N.B.: 目前判断不会多线程同时读取，故读取部分不加锁, 若发生段错误要写读取加锁函数)
    int observed_times_ = 0;  // being observed by feature matching algorithm (即为observations_的长度)

    // std::weak_ptr< MapPoint > Instead_mappoint;
    // bool is_need_instead_this_ = false;

    bool is_outlier_ = false;

    // 参见 src/sg_slam/mappoint.cpp RemoveObservation.
    // 观测到此点的激活关键帧
    // // N.B.: 个人认为最好将list改为unorder_map
    std::list< std::weak_ptr<Feature> > active_observations_; // shared_ptr<Feature> 在KeyFrame中. 
    // 被激活关键帧观测到的次数 (N.B.: 目前判断不会多线程同时读取，故读取部分不加锁, 若发生段错误要写读取加锁函数)
    int active_observed_times_ = 0;  // being observed by feature matching algorithm (即为active_observations_的长度)

    std::mutex data_mutex_;
    
public:
    MapPoint() {}

    // MapPoint(unsigned long id, Vec3 position);
    MapPoint(unsigned long id, const Vec3 &position);

    ~MapPoint() = default;

    Vec3 GetPos();

    void SetPos(const Vec3 &pos);

    void AddObservation(std::shared_ptr<Feature> feature);

    void RemoveObservation(std::shared_ptr<Feature> feature);

    void AddActiveObservation(std::shared_ptr<Feature> feature);

    void RemoveActiveObservation(std::shared_ptr<Feature> feature);

    std::list< std::weak_ptr<Feature> > GetObs();
    std::list< std::weak_ptr<Feature> > GetActiveObs();

    // factory function
    static MapPoint::Ptr CreateNewMappoint();

};


}  // namespace sg_slam 

#endif  // _SG_SLAM_MAPPOINT_H_