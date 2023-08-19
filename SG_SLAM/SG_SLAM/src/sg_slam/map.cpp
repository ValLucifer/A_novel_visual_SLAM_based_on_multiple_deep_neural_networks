//
//  Created by Lucifer on 2022/7/20.
//

#include "sg_slam/map.h"
#include "sg_slam/config.h"
#include "sg_slam/feature.h"

#include <utility>

#include <iostream>

namespace sg_slam {

// --------------------------------------------------------------------------------------------------------------
Map::Map() {
    num_active_keyframes_ = Config::Get<int>(static_cast<std::string>("Map_num_active_keyframes"));
    active_keyframe_pose_min_dis_threshold_ = Config::Get<double>(static_cast<std::string>("Map_active_keyframe_pose_min_dis_threshold"));

    std::cout << std::endl;
    std::cout << "(Map::Map()): num_active_keyframes_ = " << num_active_keyframes_ << std::endl;
    std::cout << "(Map::Map()): active_keyframe_pose_min_dis_threshold_ = " << active_keyframe_pose_min_dis_threshold_ << std::endl;
    std::cout << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
void Map::InsertKeyFrame(KeyFrame::Ptr key_frame) {
    current_keyframe_ = key_frame;

    {
        std::unique_lock<std::mutex> lck(data_mutex_); // My add
        // https://blog.csdn.net/ven21959/article/details/100621571
        // unorder_map 查询效率高
        // 若find() == end() 证明此时, 此时输入的关键帧ID不存在于map的键(key)中, 即输入关键帧不在映射集合中.
        // 即新关键帧加入
        // 激活区是全局的子集，不再全局的一定不会在激活区中.
        if(keyframes_map_.find(key_frame->id_) == keyframes_map_.end() ) {
            // 直接插入映射集合 和 激活集合
            // http://c.biancheng.net/view/7240.html
            // https://www.cnblogs.com/Nimeux/archive/2010/10/05/1844191.html
            keyframes_map_.insert( std::make_pair(key_frame->id_, key_frame) );
            active_keyframes_map_.insert( std::make_pair(key_frame->id_, key_frame) );

        } else {
            // 已有键，用新的值直接覆盖旧的值.
            keyframes_map_[key_frame->id_] = key_frame;
            active_keyframes_map_[key_frame->id_] = key_frame;
        }

        // N.B.: 在生成关键帧之前就已经将 FrameResult 加入了 frame_results_map_, 整个过程中frame_results_map_ 中的指针都不变，
        // FrameResult 的参考关键帧也不变只是参考关键帧的位姿发生改变.
        frame_results_map_[key_frame->src_frame_id_]->SetActiveKeyFlag(true);  // My add

    }

    // N.B.: 这里没有像清华的那样，再地图点中插入激活观测，有必要之后再加!!!!!


    // N.B.: 这里没有加data_mutex_锁
    // 超出激活框长度，移除特定激活关键帧(在RemoveOldKeyframe()判定移除条件，不一定是最早的关键).
    if( active_keyframes_map_.size() > num_active_keyframes_ ) {
        RemoveOldKeyframe();  
        // CleanActiveMap();
        CleanMap();
    }
}

// --------------------------------------------------------------------------------------------------------------
void Map::InsertMapPoint(MapPoint::Ptr map_point) {
    std::unique_lock<std::mutex> lck(data_mutex_); // My add
    // 参见 Map::InsertKeyFrame
    // 集合中的地图点ID不会重复.
    // 激活区是全局的子集，不再全局的一定不会在激活区中.
    // N.B.: 在关键帧生成时就加加入了对应的激活观测和全局观测，之后再减
    // N.B.: 回环后会把旧的(已经在全局中的)地图点重新设置为激活地图点故需要分开判定并插入
    // N.B.: 参见 void LoopClosing::LoopCorrectActive()

    if( landmarks_map_.find(map_point->id_) == landmarks_map_.end() ) {
        landmarks_map_.insert( std::make_pair(map_point->id_, map_point) );
    } else {
        landmarks_map_[map_point->id_] = map_point;
    }

    if( active_landmarks_map_.find(map_point->id_) == active_landmarks_map_.end()) {
        active_landmarks_map_.insert( std::make_pair(map_point->id_, map_point) );
    } else {
        active_landmarks_map_[map_point->id_] = map_point;
    }
}

// --------------------------------------------------------------------------------------------------------------
// 
void Map::InsertActiveMapPoint(MapPoint::Ptr map_point) {
    std::unique_lock<std::mutex> lck(data_mutex_); // My add

    // 参见 Map::InsertKeyFrame
    if( active_landmarks_map_.find(map_point->id_) == active_landmarks_map_.end() ) {
        active_landmarks_map_.insert( std::make_pair(map_point->id_, map_point) );
    } else {
        active_landmarks_map_[map_point->id_] = map_point;
    }
}

// --------------------------------------------------------------------------------------------------------------
void Map::InsertFrameResult(FrameResult::Ptr frame_result) {
    // 参见 Map::InsertKeyFrame
    if( frame_results_map_.find(frame_result->id_) == frame_results_map_.end() ) {
        frame_results_map_.insert( std::make_pair(frame_result->id_, frame_result) );
    } else {
        frame_results_map_[frame_result->id_] = frame_result;
    }
}

// --------------------------------------------------------------------------------------------------------------
Map::Landmarks_mapType Map::GetAllMapPoints() {
    std::unique_lock<std::mutex> lck(data_mutex_);
    return landmarks_map_;
}

// --------------------------------------------------------------------------------------------------------------
Map::Keyframes_mapType Map::GetAllKeyFrames() {
    std::unique_lock<std::mutex> lck(data_mutex_);
    return keyframes_map_;
}

// --------------------------------------------------------------------------------------------------------------
Map::Landmarks_mapType Map::GetActiveMapPoints() {
    std::unique_lock<std::mutex> lck(data_mutex_);
    return active_landmarks_map_;
}

// --------------------------------------------------------------------------------------------------------------
Map::Keyframes_mapType Map::GetActiveKeyFrames() {
    std::unique_lock<std::mutex> lck(data_mutex_);
    return active_keyframes_map_;
}

// --------------------------------------------------------------------------------------------------------------
Map::FrameResults_mapType Map::GetAllFrameResults() {
    return frame_results_map_;
}

// --------------------------------------------------------------------------------------------------------------
KeyFrame::Ptr Map::GetCurrentKeyFrame() {
    return current_keyframe_;
}

// --------------------------------------------------------------------------------------------------------------
void Map::CleanActiveMap() {

    std::unique_lock<std::mutex> lck(data_mutex_); // My add

    int cnt_active_landmark_removed = 0;
    // https://blog.csdn.net/educast/article/details/17024195
    for(auto iter = active_landmarks_map_.begin(); iter != active_landmarks_map_.end(); ) {
        if(iter->second->active_observed_times_ == 0) {
            iter = active_landmarks_map_.erase(iter);
            cnt_active_landmark_removed++;
        } else {
            ++iter;
        }
    }

    std::cout << "(Map::CleanActiveMap): Removed " << cnt_active_landmark_removed << " active landmarks " << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
void Map::CleanMap() {

    std::unique_lock<std::mutex> lck(data_mutex_); // My add
    int cnt_active_landmark_removed = 0;
    int cnt_landmark_removed = 0;
    // 参见 Map::CleanActiveMap
    for(auto iter = active_landmarks_map_.begin(); iter != active_landmarks_map_.end(); ) {
        if(iter->second->active_observed_times_ == 0) {
            iter = active_landmarks_map_.erase(iter);
            cnt_active_landmark_removed++;
        } else {
            ++iter;
        }
    }

    for(auto iter = landmarks_map_.begin(); iter != landmarks_map_.end(); ) {
        if(iter->second->observed_times_ == 0) {
            iter = landmarks_map_.erase(iter);
            cnt_landmark_removed++;
        } else {
            ++iter;
        }
    }

    std::cout << "Map::CleanMap: Removed " << cnt_active_landmark_removed << " active landmarks. " << std::endl;
    std::cout << "Map::CleanMap: Removed " << cnt_landmark_removed << " landmarks. " << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
void Map::RemoveMapPoint(MapPoint::Ptr map_point) {
    std::unique_lock<std::mutex> lck(data_mutex_); // My add

    unsigned long mpId = map_point->id_;

    // N.B.: 与清华的有所不同!!!!!
    // delete from all mappoints
    if( landmarks_map_.find(mpId) != landmarks_map_.end() ) {
        landmarks_map_.erase(mpId);
    }

    // delete from active mappoints
    if( active_landmarks_map_.find(map_point->id_) != active_landmarks_map_.end()) {
        active_landmarks_map_.erase(mpId);
    }
}


// --------------------------------------------------------------------------------------------------------------
// private function
void Map::RemoveOldKeyframe() {
    std::unique_lock<std::mutex> lck(data_mutex_); // My add

    // 20220921 改
    if( current_keyframe_ == nullptr ) {
        // N.B.: 当前没有插入关键帧，根据调用情况这个判断应该是多余的!!!!
        return;
    }

    
    // 寻找与当前帧最近和最远的两个关键帧(使用Tcw的差异度量)
    double max_dis = 0, min_dis = 9999.0;
    unsigned long max_kf_id = 0, min_kf_id = 0;
    auto Twc = current_keyframe_->Pose().inverse();  // current

    for(auto &kf : active_keyframes_map_) {
        if(kf.second == current_keyframe_) {
            // 不和自己比
            continue;
        }

        // 先转换为李代数(向量)，再计算模(二范数).
        auto dis = (kf.second->Pose() * Twc).log().norm();
        if( dis > max_dis ) {
            max_dis = dis;
            max_kf_id = kf.first;
        }

        if( dis < min_dis ) {
            min_dis = dis;
            min_kf_id = kf.first;
        }
    }


    // // decide which kf to be removed
    KeyFrame::Ptr active_keyframe_to_remove = nullptr;
    if( min_dis < active_keyframe_pose_min_dis_threshold_ ) {
        // https://www.nhooo.com/note/qa03ff.html
        // 如果存在很近(李代数模度量)的帧，优先删掉最近的(如相机长时间不动等)
        // N.B.: 与十四讲不同，个人认为在active_keyframes_map_索引会更好，
        // N.B.: 在keyframes_map_索引也是一样的，因为存储时key都是相同的关键帧ID
        active_keyframe_to_remove = active_keyframes_map_.at(min_kf_id);
        // active_keyframe_to_remove = keyframes_map_.at(min_kf_id);
    } else {
        // 删掉最远的
        active_keyframe_to_remove = active_keyframes_map_.at(max_kf_id);
        // active_keyframe_to_remove = keyframes_map_.at(max_kf_id);
    }
    
    // // decide which kf to be removed

    std::cout << "(Map::RemoveOldKeyframe): remove active keyframe " << active_keyframe_to_remove->id_ << std::endl;

    active_keyframes_map_.erase(active_keyframe_to_remove->id_);
    
    // My add
    // N.B.: 参见 Map::InsertKeyFrame
    frame_results_map_[active_keyframe_to_remove->src_frame_id_]->SetActiveKeyFlag(false);
    // N.B.: 使用auto， 也要包含对应的数据类型声明头文件
    for(auto &feat : active_keyframe_to_remove->mvpfeatures) {
        // N.B.: 为空时，若操作会产生段错误 !!!!!!
        if(feat == nullptr) {
            continue;
        }
        
        auto mp = feat->map_point_.lock();
        
        // N.B.: 测试一下
        // N.B.: shared_ptr 可与 nullptr做判定, weak_ptr不行.
        if(mp) {
            mp->RemoveActiveObservation(feat);
        }
        
    }

}

}  // namespace sg_slam
