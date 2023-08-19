//
//  Created by Lucifer on 2022/7/6.
//

#include "sg_slam/mappoint.h"
#include "sg_slam/feature.h"  // N.B.: 需要调用Feature的成员变量/函数， 光forward declare没有用，需要包含对应头文件

namespace sg_slam {

// --------------------------------------------------------------------------------------------------------------
MapPoint::MapPoint(unsigned long id, const Vec3 &position) 
    : id_(id), pos_(position) {

}

// --------------------------------------------------------------------------------------------------------------
Vec3 MapPoint::GetPos() {
    std::unique_lock< std::mutex > lck(data_mutex_);
    return pos_;
}

// --------------------------------------------------------------------------------------------------------------
void MapPoint::SetPos(const Vec3 &pos) {
    std::unique_lock< std::mutex > lck(data_mutex_);
    pos_ = pos;
}

// --------------------------------------------------------------------------------------------------------------
void MapPoint::AddObservation(std::shared_ptr<Feature> feature) {
    std::unique_lock< std::mutex > lck(data_mutex_);
    // N.B.: 个人认为最好和移除一样加判断，若已存在则不加, 用unorder_map/unorder_set结构比较好
    observations_.push_back( feature );
    observed_times_++;
}

// --------------------------------------------------------------------------------------------------------------
// N.B.: 仅发生点融合时调用
void MapPoint::RemoveObservation(std::shared_ptr<Feature> feature) {
    std::unique_lock< std::mutex > lck(data_mutex_);
    for(auto iter = observations_.begin(); iter != observations_.end(); iter++) {
        // 调用lock()获取对应的shared_ptr.
        if(iter->lock() == feature) {
            observations_.erase(iter);
            // feature->map_point_.reset();  // N.B.: 是weak_ptr的对象变为空指针，类似默认构造函数(与shared_ptr, unique_ptr的reset不同, weak_ptr的reset不能带参数).
            feature->EraseMapPoint();  // N.b.: 
            observed_times_--;  // N.B.: 漏加了.
            break;
        }
    }

    // N.B.: 由于多线程运行这里最好加上一个小于零的判断.
    if(observed_times_ < 0) {
        observed_times_ = 0;
    }
}

// --------------------------------------------------------------------------------------------------------------
void MapPoint::AddActiveObservation(std::shared_ptr<Feature> feature) {
    std::unique_lock< std::mutex > lck(data_mutex_);
    // N.B.: 个人认为最好和移除一样加判断，若已存在则不加, 用unorder_map/unorder_set结构比较好
    active_observations_.push_back( feature );
    active_observed_times_++;
}

// --------------------------------------------------------------------------------------------------------------
// N.B.: 执行原本十四讲的MapPoint::RemoveObservation功能, 与清华的相同.
void MapPoint::RemoveActiveObservation(std::shared_ptr<Feature> feature) {
    std::unique_lock< std::mutex > lck(data_mutex_);
    
    if(feature == nullptr) {
        return;
    }

    for(auto iter = active_observations_.begin(); iter != active_observations_.end(); iter++) {
        // 调用lock()获取对应的shared_ptr.
        if(iter->lock() == feature) {
            active_observations_.erase(iter);
            // N.B.: 这是删除全局关联的不可加.
            // feature->map_point_.reset();
            active_observed_times_--;
            break;
        }
    }
    // std::cout << "(MapPoint::RemoveActiveObservation): ++++++++++++++++++2+++++++++++++++++" << std::endl;

    // N.B.: 由于多线程运行这里最好加上一个小于零的判断.
    if(active_observed_times_ < 0) {
        active_observed_times_ = 0;
    }
}




// --------------------------------------------------------------------------------------------------------------
std::list< std::weak_ptr<Feature> > MapPoint::GetObs() {
    std::unique_lock< std::mutex > lck(data_mutex_);
    return observations_;
}

std::list< std::weak_ptr<Feature> > MapPoint::GetActiveObs() {
    std::unique_lock< std::mutex > lck(data_mutex_);
    return active_observations_;
}

// --------------------------------------------------------------------------------------------------------------
MapPoint::Ptr MapPoint::CreateNewMappoint() {
    static long factory_id = 0;
    MapPoint::Ptr new_mappoint(new MapPoint);
    new_mappoint->id_ = factory_id++;
    new_mappoint->is_outlier_ = false;  // N.B.: 自加
    return new_mappoint;
}

}  // namespace sg_slam
