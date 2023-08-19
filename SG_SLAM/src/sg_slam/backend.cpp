//
//  Created by Lucifer on 2022/7/29.
//

#include "sg_slam/backend.h"
#include <functional>  // std::bind  // 参见 sg_slam/viewer.h #include <thread>
#include <map>
#include <unistd.h>  // usleep // 参见 sg_slam/viewer.cpp

#include "sg_slam/g2o_types.h"
#include "sg_slam/algorithm.h"

#include "sg_slam/common_include.h"
#include "sg_slam/config.h"
#include "sg_slam/feature.h"

#include <g2o/core/block_solver.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>  // 2022100401
#include <g2o/core/optimization_algorithm_gauss_newton.h>  // 2022100401
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>

#include <g2o/types/sba/edge_project_stereo_xyz.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sba/edge_project_xyz.h>

#include <iostream>

namespace sg_slam {

// --------------------------------------------------------------------------------------------------------------
Backend::Backend() {
    m_backend_running_.store(true);
    m_request_pause_.store(false);
    m_has_paused_.store(false);
    m_is_stop_flag.store(false);

    // 参见 sg_slam/viewer.h 
    m_backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
    m_edge_robust_kernel_chi2_th_ = Config::Get<double>(static_cast<std::string>("Backend_edge_robust_kernel_chi2_th"));
    m_optimized_iter_ = Config::Get<int>(static_cast<std::string>("Backend_optimized_iter"));
    m_max_iteration_ = Config::Get<int>(static_cast<std::string>("Backend_max_iteration"));

    std::cout << std::endl;
    std::cout << "(Backend::Backend()): m_edge_robust_kernel_chi2_th_ = " << m_edge_robust_kernel_chi2_th_ << std::endl;
    std::cout << "(Backend::Backend()): m_optimized_iter_ = " << m_optimized_iter_ << std::endl;
    std::cout << "(Backend::Backend()): m_max_iteration_ = " << m_max_iteration_ << std::endl;
    std::cout << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
void Backend::SetCameras(Camera::Ptr left, Camera::Ptr right) {
    m_cam_left_ = left;
    m_cam_right_ = right;
}

// --------------------------------------------------------------------------------------------------------------
void Backend::SetMap(Map::Ptr map) {
    m_map_ = map;
}

// --------------------------------------------------------------------------------------------------------------
void Backend::InsertProcessKeyframe() {
    std::unique_lock< std::mutex > lck(m_backend_cache_mutex_);
    assert(m_map_ != nullptr);
    unsigned long current_keyframe_id = m_map_->GetCurrentKeyFrame()->id_;
    m_process_keyframes_id_list.push_back(current_keyframe_id);
}

// --------------------------------------------------------------------------------------------------------------
bool Backend::CheckNeedProcessKeyframe() {
    std::unique_lock< std::mutex > lck(m_backend_cache_mutex_);
    return (!m_process_keyframes_id_list.empty());
}

// --------------------------------------------------------------------------------------------------------------
void Backend::Stop() {
    m_backend_running_.store(false);
    m_backend_thread_.join();
    std::cout << "Stop Backend!" << std::endl;
    m_is_stop_flag.store(true);
}

// --------------------------------------------------------------------------------------------------------------
void Backend::RequestPause() {
    m_request_pause_.store(true);
}

// --------------------------------------------------------------------------------------------------------------
bool Backend::IfHasPaused() {
    return ((m_request_pause_.load()) && (m_has_paused_.load()));
}

// --------------------------------------------------------------------------------------------------------------
void Backend::Resume() {
    m_request_pause_.store(false);
}

// private function
// --------------------------------------------------------------------------------------------------------------
void Backend::BackendLoop() {
    // N.B.: 加上 CheckNeedProcessKeyframe(), 防止在停止后端优化线程时，缓存区还有没处理完的关键帧.
    while(m_backend_running_.load() || CheckNeedProcessKeyframe()) {
        if(CheckNeedProcessKeyframe()) {
            Map::Keyframes_mapType active_kfs;
            Map::Landmarks_mapType active_landmarks;
            
            {
                std::unique_lock< std::mutex > lck(m_backend_cache_mutex_);
                // unsigned long current_keyframe_id = m_process_keyframes_id_list.front();
                // active_kfs = m_process_active_keyframes_map_[current_keyframe_id];
                // active_landmarks = m_process_active_landmarks_map_[current_keyframe_id];
                m_process_keyframes_id_list.pop_front();
                // m_process_active_keyframes_map_.erase(current_keyframe_id);
                // m_process_active_landmarks_map_.erase(current_keyframe_id);
            }

            active_kfs = m_map_->GetActiveKeyFrames();
            active_landmarks =  m_map_->GetActiveMapPoints();

            // Optimize(active_kfs, active_landmarks);
            OptimizeStereo(active_kfs, active_landmarks);
        }

        // N.B.: 这里与清华工程的暂停逻辑不同. !!!!! (实验验证效果)
        // if the loopclosing thread asks backend to pause, 
        // this will make sure that the backend will pause in this position.
        while(m_request_pause_.load()) {
            m_has_paused_.store(true);
            // usleep(1000);  // N.B.: 清华工程有加，这里个人认为去掉更好，实验验证.
        }
        m_has_paused_.store(false);
    }

    std::cout << "Stop Backend::BackendLoop()!" << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
void Backend::Optimize(Map::Keyframes_mapType &keyframes_map, Map::Landmarks_mapType &landmarks_map) {

    
    if(keyframes_map.size() <= 1) {
        std::cout << std::endl;
        std::cout << "Backend::Optimize: active keyframes has only one keyframe! " << std::endl;
        std::cout << std::endl;
        return;
    }

    // setup g2o
    // solver for BA/3D SLAM 
    // pose dimension is 6 (se3), landmark dimension is 3.
    using BlockSolverType = g2o::BlockSolver_6_3;
    // 线性求解器类型(稀疏)
    using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;

    // use LM (N.B.: 可以换优化搜索算法)
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>( g2o::make_unique<LinearSolverType>() ) );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm( solver );


    // vertex of pose (active), use keyframe id
    std::unordered_map< unsigned long, VertexPose* > vertices;
    unsigned long max_kf_id = 0;
    for(auto &keyframe : keyframes_map) {

        auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
        vertex_pose->setId(kf->id_);  // 设置顶点 ID.
        vertex_pose->setEstimate(kf->Pose());  // 设置初值(迭代初值T0)
        
        // N.B.: My add, 判断关键帧ID是否为0，即是否是第一帧(整个系统的参考系)，若是第一帧则固定不优化
        // N.B.: 当优化器的一种顶点只有一个时, 即当激活关键帧集合中只有一个第0帧时，不能固定，否则g2o会报错.
        if(kf->id_ == 0) {
            vertex_pose->setFixed(true);
        }

        optimizer.addVertex(vertex_pose);  // 在优化器中添加顶点(待优化变量)
        
        if(kf->id_ > max_kf_id) {
            max_kf_id = kf->id_;
        }

        vertices.insert( {kf->id_, vertex_pose} );
        // vertices.insert( {keyframe.first, vertex_pose} ); // N.B.: 改成这个会更好!!!!!
    }
    // 路标顶点，使用路标id索引 + (max_kf_id+1), 加1是因为从0开始计数，若不加1，最后一个pose顶点与第一个路标顶点ID相同，会出错.
    std::map< unsigned long, VertexXYZ* > vertices_landmarks;

    // K((左)内参) 和 左右外参
    Mat33 K = m_cam_left_->K();
    SE3 left_extrinsic = m_cam_left_->pose();
    SE3 right_extrinsic = m_cam_right_->pose();

    // edges, id从1开始
    int index = 1;
    // 自由度为2， 标准差为1个像素(已白化或假定测量特征点位置标准差为1个像素), 置信度为95%(alpha=0.05)的分位数(分位点);
    // double chi2_th = 5.991;  // robust kernel 阈值 (可调)
    std::map< EdgeProjection*, Feature::Ptr> edges_and_features;

    // N.B.: 没有加清华的将点固定的条件，个人认为不合适.
    for(auto &landmark : landmarks_map) {
        if(landmark.second->is_outlier_) {
            continue;
        }

        unsigned long landmark_id = landmark.second->id_;

        // N.B.: 个人改为激活观测，十四讲中的观测实则为激活观测.
        // auto observations = landmark.second->GetObs();
        auto active_observations = landmark.second->GetActiveObs();
        for(auto &obs : active_observations) {
            // N.B.: 观测为nullptr, 在本系统中不太可能发生，之后可以去掉这个判断试一试!!!!!
            if(obs.lock() == nullptr) {
                continue;
            }

            auto feat = obs.lock();
            // auto kf = feat->KeyFrame_.lock();
            // N.B.: 验证观测是否在激活区内，个人认为加不加无影响，因为本身是在激活观测中取的
            // assert(keyframes_map.find(kf->id_) != keyframes_map.end());

            // N.B.: feat->KeyFrame_.lock() == nullptr, 在本系统中不太可能发生，之后可以去掉这个判断试一试!!!!!
            if(feat->is_outlier_ || feat->KeyFrame_.lock() == nullptr) {
                continue;
            }

            auto kf = feat->KeyFrame_.lock();
            // N.B.: 验证观测是否在激活区内，个人认为加不加无影响，因为本身是在激活观测中取的
            // assert(keyframes_map.find(kf->id_) != keyframes_map.end());

            EdgeProjection *edge = nullptr;
            if(feat->is_on_left_image_) {
                edge = new EdgeProjection(K, left_extrinsic);
            } else {
                edge = new EdgeProjection(K, right_extrinsic);
            }

            // 如果landmark还没有被加入优化，则新加一个顶点
            // N.B.: 放在激活观测循环内的好处，若此点是激活点，但没有一个通过要求的激活观测(虽然可能性很低)，这个点不会加入到待优化顶点中，
            // 不用在加入待优化顶点集合后再，对其进行固定，因为没有边(约束), 虽然会增加判断次数.
            if(vertices_landmarks.find(landmark_id) == vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ;
                // VertexXYZ *v = new VertexXYZ();  //N.B.: 测试一下是否有差别!!!!!

                v->setEstimate( landmark.second->GetPos() );
                v->setId(landmark_id + max_kf_id + 1);
                // g2o在BA中需要手动设置待Marg(边缘化)的顶点
                v->setMarginalized(true);
                vertices_landmarks.insert( {landmark_id, v} );
                // vertices_landmarks.insert( {landmark_id, v} );  // N.B.: 改成这个会更好!!!!!
                optimizer.addVertex(v);
            }

            edge->setId( index );
            edge->setVertex(0, vertices.at(kf->id_));  // keyframe pose
            edge->setVertex(1, vertices_landmarks.at(landmark_id));  // landmark
            edge->setMeasurement( toVec2(feat->position_.pt) );
            edge->setInformation( Mat22::Identity() );  // 信息矩阵默认设为单位矩阵, 故测量误差最好白化，或者根据方差设置信息矩阵.

            auto rk = new g2o::RobustKernelHuber();  // robust kernel 十四讲(第二版) P252
            rk->setDelta(m_edge_robust_kernel_chi2_th_);
            edge->setRobustKernel(rk);  // N.B.: 漏写了，!!!!!!
            edges_and_features.insert( {edge, feat} );  // feat 等价于 feature

            optimizer.addEdge(edge);

            index++;
        }
    }

    // do optimization and eliminate(排除) the outliers
    // 这里的逻辑和清华的类似，十四讲的有错没有优化.
    // optimizer.initializeOptimization();
    // optimizer.optimize(10); // 每一次优化迭代10次， 可写在配置文件上(外化)

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    
    double chi2_th = m_edge_robust_kernel_chi2_th_; // N.B.: 可能会变，但目前假定不变.!!!!!
    // 最大优化次数5次， 可写在配置文件上(外化)
    // while(iteration < 5) {
    while(iteration < m_max_iteration_) {
        optimizer.initializeOptimization();
        // optimizer.optimize(10); 
        optimizer.optimize(m_optimized_iter_);
        cnt_outlier = 0;
        cnt_inlier = 0;

        // determine if we want to adjust the outlier threshold (阈值也可不调整, 再进行一次优化)
        // N.B.: 只统计内点率，判断是否再次优化(在次优化前没有剔除外点，若精度不高可以在此改进)!!!!!
        for(auto &ef : edges_and_features) {
            if(ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }

        double inlier_ratio = static_cast<double>(cnt_inlier) / static_cast<double>(cnt_inlier + cnt_outlier);
        std::cout << "(Backend::Optimize:) inlier_ratio = " << inlier_ratio << std::endl;
        if(inlier_ratio > 0.5) {
            break; // 满足内点要求，提前跳出优化(不一定执行5次) 
        } else {
            // chi2_th *= 2;
            iteration++;
        }
    }

    // N.B.: 整体优化结束后剔除外点(地图点).
    for(auto &ef : edges_and_features) {
        if(ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;  // feature
            // remove the observation(同时移除激活观测和全局观测，因为关联的特征是外点)
            // auto mp = ef.second->map_point_.lock();
            auto mp = ef.second->GetMapPoint();
            mp->RemoveActiveObservation(ef.second);
            mp->RemoveObservation(ef.second);

            // N.B.: 不在特征中删除地图点，而是在对应的关键帧中删除特征.


            // if the mappoint has no good observation, then regard it as a outlier. 
            // It will be deleted later.
            // N.B.: 不用像清华那样设置地图清除表示，当全局观测为0时，下一次向地图插入关键帧时(前端执行)，会自动清除当时全局观测为0的点。
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    std::cout << "(Backend::Optimize:) Outlier/Inlier in optimization: " << cnt_outlier << "/" << cnt_inlier;
    
    // Set pose and landmark position
    for(auto &v : vertices) {
        keyframes_map.at(v.first)->SetPose(v.second->estimate());
    }
    for(auto &v : vertices_landmarks) {
        landmarks_map.at(v.first)->SetPos(v.second->estimate());
    }
}

// --------------------------------------------------------------------------------------------------------------
void Backend::OptimizeStereo(Map::Keyframes_mapType &keyframes_map, Map::Landmarks_mapType &landmarks_map) {

    if(keyframes_map.size() <= 1) {
        std::cout << std::endl;
        std::cout << "Backend::Optimize: active keyframes has only one keyframe! " << std::endl;
        std::cout << std::endl;
        return;
    }

    // setup g2o
    // solver for BA/3D SLAM 
    // pose dimension is 6 (se3), landmark dimension is 3.
    using BlockSolverType = g2o::BlockSolver_6_3;
    // 线性求解器类型(稀疏)
    using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;

    // use LM (N.B.: 可以换优化搜索算法)
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>( g2o::make_unique<LinearSolverType>() ) );
    
    // 20221004
    /**
    auto solver = new g2o::OptimizationAlgorithmDogleg(
        g2o::make_unique<BlockSolverType>( g2o::make_unique<LinearSolverType>() ));
    **/

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm( solver );

    // vertex of pose (active), use keyframe id
    std::unordered_map< unsigned long, g2o::VertexSE3Expmap* > vertices;
    unsigned long max_kf_id = 0;
    for(auto &keyframe : keyframes_map) {
        auto kf = keyframe.second;
        g2o::VertexSE3Expmap *vertex_pose = new g2o::VertexSE3Expmap();
        auto kf_pose = kf->Pose();
        auto kf_pose_SE3Quat = g2o::SE3Quat(kf_pose.rotationMatrix(), kf_pose.translation());
        vertex_pose->setId(kf->id_);  // 设置顶点 ID.
        vertex_pose->setEstimate(kf_pose_SE3Quat);  // 设置初值(迭代初值T0)

        // N.B.: My add, 判断关键帧ID是否为0，即是否是第一帧(整个系统的参考系)，若是第一帧则固定不优化
        // N.B.: 当优化器的一种顶点只有一个时, 即当激活关键帧集合中只有一个第0帧时，不能固定，否则g2o会报错.
        // 20221012 改
        /**
        if(kf->id_ == 0) {
            vertex_pose->setFixed(true);
        }
        **/

        optimizer.addVertex(vertex_pose);  // 在优化器中添加顶点(待优化变量)

        if(kf->id_ > max_kf_id) {
            max_kf_id = kf->id_;
        }

        vertices.insert( {kf->id_, vertex_pose} );
    }

    // 路标顶点，使用路标id索引 + (max_kf_id+1), 加1是因为从0开始计数，若不加1，最后一个pose顶点与第一个路标顶点ID相同，会出错.
    std::unordered_map< unsigned long, g2o::VertexPointXYZ* > vertices_landmarks;

    // K((左)内参) 和 左右外参
    auto left_fx = m_cam_left_->fx_;
    auto left_fy = m_cam_left_->fy_;
    auto left_cx = m_cam_left_->cx_;
    auto left_cy = m_cam_left_->cy_;
    auto rl_bf = m_cam_right_->fx_baseline_;

    std::cout << std::endl;
    std::cout << "(Backend::OptimizeStereo()): rl_bf = " << rl_bf << std::endl;
    std::cout << std::endl;

    // edges, id从1开始
    int index = 1;
    // 自由度为3， 标准差为1个像素(已白化或假定测量特征点位置标准差为1个像素), 置信度为95%(alpha=0.05)的分位数(分位点);
    const float thHuberStereo = sqrt(7.815); // robust kernel 阈值 (可调)
    const float thHuber2d = sqrt(m_edge_robust_kernel_chi2_th_);  // 自由度为2
    std::unordered_map< g2o::EdgeStereoSE3ProjectXYZ*, Feature::Ptr > edges_and_features;
    std::unordered_map< g2o::EdgeSE3ProjectXYZ*, Feature::Ptr > edges2d_and_features;

    for(auto &landmark : landmarks_map) {
        if(landmark.second->is_outlier_) {
            continue;
        }

        unsigned long landmark_id = landmark.second->id_;

        auto active_observations = landmark.second->GetActiveObs();
        for(auto &obs : active_observations) {
            // N.B.: 观测为nullptr, 在本系统中不太可能发生，之后可以去掉这个判断试一试!!!!!
            if(obs.lock() == nullptr) {
                continue;
            }

            auto feat = obs.lock();

            // N.B.: feat->KeyFrame_.lock() == nullptr, 在本系统中不太可能发生，之后可以去掉这个判断试一试!!!!!
            if(feat->is_outlier_ || feat->KeyFrame_.lock() == nullptr) {
                continue;
            }

            // N.B.: 防止重复添加同一个点顶点
            if(vertices_landmarks.find(landmark_id) == vertices_landmarks.end()) {
                g2o::VertexPointXYZ *v = new g2o::VertexPointXYZ();
                v->setEstimate( landmark.second->GetPos() );
                v->setId(landmark_id + max_kf_id + 1);
                // g2o在BA中需要手动设置待Marg(边缘化)的顶点
                v->setMarginalized(true);
                vertices_landmarks.insert( {landmark_id, v} );
                optimizer.addVertex(v);
            }

            // 修改 20220914
            auto kf = feat->KeyFrame_.lock();

            if(feat->kpR_u_ > -1) {
                g2o::EdgeStereoSE3ProjectXYZ* edge = new g2o::EdgeStereoSE3ProjectXYZ();
                edge->setId( index );
                edge->setVertex(0, vertices_landmarks.at(landmark_id));  // landmark
                edge->setVertex(1, vertices.at(kf->id_));  // keyframe pose
                Vec3 StereoMeasure;
                StereoMeasure = toVec3(feat->position_.pt, feat->kpR_u_);
                edge->setMeasurement(StereoMeasure);
                edge->setInformation( Mat33::Identity() );  // 信息矩阵默认设为单位矩阵, 故测量误差最好白化，或者根据方差设置信息矩阵.
                auto rk = new g2o::RobustKernelHuber();
                rk->setDelta(thHuberStereo);
                edge->setRobustKernel(rk); 
                edge->fx = left_fx;
                edge->fy = left_fy;
                edge->cx = left_cx;
                edge->cy = left_cy;
                edge->bf = rl_bf;
                edges_and_features.insert( {edge, feat} );  // feat 等价于 feature
                optimizer.addEdge(edge);
            } else {
                g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
                edge->setId( index );
                edge->setVertex(0, vertices_landmarks.at(landmark_id));  // landmark
                edge->setVertex(1, vertices.at(kf->id_));  // keyframe pose
                Vec2 measure2d;
                measure2d = toVec2(feat->position_.pt);
                edge->setMeasurement(measure2d);
                edge->setInformation( Mat22::Identity() );
                auto rk = new g2o::RobustKernelHuber(); 
                rk->setDelta(thHuber2d);
                edge->setRobustKernel(rk);
                edge->fx = left_fx;
                edge->fy = left_fy;
                edge->cx = left_cx;
                edge->cy = left_cy;
                edges2d_and_features.insert( {edge, feat} );
                optimizer.addEdge(edge);
            }
            index++;
        }
    }

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;

    double chi2_th = 7.815;
    double chi2_2d_th = m_edge_robust_kernel_chi2_th_;

    while(iteration < m_max_iteration_) {
        optimizer.initializeOptimization();
        optimizer.optimize(m_optimized_iter_);
        cnt_outlier = 0;
        cnt_inlier = 0;

        for(auto &ef : edges_and_features) {
            if(ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }

        // My add 20220914
        for(auto &ef : edges2d_and_features) {
            if(ef.first->chi2() > chi2_2d_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }

        double inlier_ratio = static_cast<double>(cnt_inlier) / static_cast<double>(cnt_inlier + cnt_outlier);
        std::cout << "(Backend::OptimizeStereo:) inlier_ratio = " << inlier_ratio << std::endl;
        if(inlier_ratio > 0.5) {
            break; // 满足内点要求，提前跳出优化(不一定执行5次) 
        } else {
            // chi2_th *= 2;
            iteration++;
        }
    }

    // N.B.: 整体优化结束后剔除外点(地图点).
    for(auto &ef : edges_and_features) {
        if(ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;  // feature
            // remove the observation(同时移除激活观测和全局观测，因为关联的特征是外点)
            // auto mp = ef.second->map_point_.lock();
            auto mp = ef.second->GetMapPoint();
            mp->RemoveActiveObservation(ef.second);
            mp->RemoveObservation(ef.second);

            // N.B.: 不在特征中删除地图点，而是在对应的关键帧中删除特征.


            // if the mappoint has no good observation, then regard it as a outlier. 
            // It will be deleted later.
            // N.B.: 不用像清华那样设置地图清除表示，当全局观测为0时，下一次向地图插入关键帧时(前端执行)，会自动清除当时全局观测为0的点。
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    // My add 20220914
    for(auto &ef : edges2d_and_features) {
        if(ef.first->chi2() > chi2_2d_th) {
            ef.second->is_outlier_ = true;  // feature
            auto mp = ef.second->GetMapPoint();
            mp->RemoveActiveObservation(ef.second);
            mp->RemoveObservation(ef.second);
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    std::cout << "(Backend::OptimizeStereo:) Outlier/Inlier in optimization: " << cnt_outlier << "/" << cnt_inlier;

    // Set pose and landmark position
    for(auto &v : vertices) {
        auto kf_pose_SE3Quat = v.second->estimate();
        auto kf_q = kf_pose_SE3Quat.rotation();
        auto kf_t = kf_pose_SE3Quat.translation();
        auto kf_pose = SE3(kf_q, kf_t);

        keyframes_map.at(v.first)->SetPose(kf_pose);
    }
    for(auto &v : vertices_landmarks) {
        landmarks_map.at(v.first)->SetPos(v.second->estimate());
    }
}

}  // namespace sg_slam
