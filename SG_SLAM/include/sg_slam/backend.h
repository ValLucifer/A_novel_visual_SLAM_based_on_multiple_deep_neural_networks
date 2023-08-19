//
//  Created by Lucifer on 2022/7/29.
//

#ifndef _SG_SLAM_BACKEND_H_
#define _SG_SLAM_BACKEND_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#include <memory>  // 智能指针
#include <mutex>  // C++14多线程中的<mutex>提供了多种互斥操作，可以显式避免数据竞争。
#include <list>
#include <unordered_map> 

#include <thread>  // 多线程 C++11/14 参见 sg_slam/viewer.h

// https://zh.cppreference.com/w/cpp/thread/condition_variable
// #include <condition_variable>  

// C++11起提供了atomic，可以使用它定义一个原子(不可分割)类型
// https://blog.csdn.net/nihao_2014/article/details/124908784
// 为什么要定义一个原子类型？
// 举个例子，int64_t类型，在32位机器上为非原子操作。更新时该类型的值时，需要进行两步操作（高32位、低32位）。如果多线程操作该类型的变量，且在操作时未加锁，可能会出现读脏数据的情况。
// 解决该问题的话，加锁，或者提供一种定义原子类型的方法。
#include <atomic>  

#include "sg_slam/camera.h"
#include "sg_slam/map.h"

namespace sg_slam {

/**
 * @brief backend(后端)
 * 有单独优化线程， 前端将新关键帧插入地图，地图再将新插入的关键帧和对应激活区信息(激活关键帧和激活地图点)插入后端缓存区，后端缓存区非空时启动优化.
 * 
 */
class Backend 
{
public:  
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h
    using this_type = Backend;  // 本类别名
    using Ptr = std::shared_ptr<Backend>;  // 本类共享/弱指针别名

    std::atomic<bool> m_is_stop_flag;  // N.B.: 结束线程时， 其他线程查看，此线程是否已经停止了(自加)!!!!! 防止误操作空指针

    // 构造函数中启动优化线程并挂起
    Backend();
    ~Backend() = default;

    // 设置左右目相机，用于获取相机的内外参
    // N.B.: 双目及双目以上.
    void SetCameras(Camera::Ptr left, Camera::Ptr right);

    // 设置地图
    // void SetMap(std::shared_ptr<Map> map) {
    void SetMap(Map::Ptr map);

    // N.B.: 本系统中不使用触发方式启动优化
    // 触发地图更新, 启动优化
    // void UpdateMap();
    // N.B.: process_keyframe_id: 触发优化时，要处理的当前关键帧(相对于触发时刻), 
    // 当插入关键帧间隔过小时，来不及处理完一帧(批)优化，又来一帧时的缓存，缓冲空间可在Map中定义.
    // void UpdateMap(unsigned long process_keyframe_id);

    // N.B.: My add
    // 插入缓存数据，关键帧插入地图后插入.
    // 前端调用.
    void InsertProcessKeyframe();

    // N.B.: My add
    // 检测缓存队列是否不为空，以此来启动优化.
    bool CheckNeedProcessKeyframe();

    // 关闭后端线程
    void Stop();

    // ask the backend thread to pause (请求后端线程暂停) (回环线程调用)
    void RequestPause();

    // return true if the backend thread has paused. (回环线程调用)
    bool IfHasPaused(); 

    // ask the backend thread to resume running (回环线程调用)
    // 请求后端线程恢复运行
    void Resume();

private:
    // 后端线程执行函数
    void BackendLoop();

    // 对给定关键帧和路标点进行优化
    void Optimize(Map::Keyframes_mapType &keyframes_map, Map::Landmarks_mapType &landmarks_map);

    // My add 20220913 (根据ORB-SLAM2修改的)
    void OptimizeStereo(Map::Keyframes_mapType &keyframes_map, Map::Landmarks_mapType &landmarks_map);

    // std::shared_ptr<Map> map_;
    Map::Ptr m_map_;  // 为了与地图交互
    std::thread m_backend_thread_;  // N.B.: 多线程处理句柄(handle) 参见 sg_slam/viewer.h
    // std::mutex m_backend_data_mutex_;

    // condition_variable 类是同步原语，能用于阻塞一个线程，或同时阻塞多个线程，直至另一线程修改共享变量（条件）并通知 condition_variable 。 
    // https://zh.cppreference.com/w/cpp/thread/condition_variable
    // std::condition_variable m_map_update_;  // C++11起.

    // http://wjhsh.net/pjl1119-p-9715815.html
    // C++11之 std::atomic （不用锁实现线程互斥）
    std::atomic<bool> m_backend_running_;  // ( 一个原子的布尔类型，可支持两种原子操作 )并发编程中的一种原子操作 C++11
    // N.B.: 与回环交互.
    // // N.B.: 原清华工程的 _mbRequestPause
    std::atomic<bool> m_request_pause_;
    // // N.B.: 原清华工程的 _mbHasPaused
    std::atomic<bool> m_has_paused_;

    Camera::Ptr m_cam_left_ = nullptr;
    Camera::Ptr m_cam_right_ = nullptr;

    // N.B.: My Add (关键帧和激活区缓存数据库)
    // 待处理关键帧ID队列
    std::list< unsigned long > m_process_keyframes_id_list;  // 待处理关键帧ID队列
    // 待处理激活区关键帧映射集合，键为处理时刻的当前关键帧ID，值为理时刻的当前关键帧对应的激活区关键帧映射集合.
    // std::unordered_map<unsigned long, Map::Keyframes_mapType> m_process_active_keyframes_map_; 
    // 待处理激活区地图点映射集合，键为处理时刻的当前关键帧ID，值为理时刻的当前关键帧对应的激活区地图点映射集合.
    // std::unordered_map<unsigned long, Map::Landmarks_mapType> m_process_active_landmarks_map_;
    std::mutex m_backend_cache_mutex_;  //后端缓存锁. 

    // N.B.: My Add
    // Settings
    // // 自由度为2， 标准差为1个像素(已白化或假定测量特征点位置标准差为1个像素), 置信度为95%(alpha=0.05)的分位数(分位点);
    double m_edge_robust_kernel_chi2_th_ = 5.991;

    // N.B.: 每次优化时，优化器的迭代次数.
    int m_optimized_iter_ = 10;  // N.B.: 每次优化时，优化器的迭代次数.
    int m_max_iteration_ = 5;  // N.B.: 最大优化次数.


};  // class Backend

}  // namespace sg_slam

#endif  // _SG_SLAM_BACKEND_H_
