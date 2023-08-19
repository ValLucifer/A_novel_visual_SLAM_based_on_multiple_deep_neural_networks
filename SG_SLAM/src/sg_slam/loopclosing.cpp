//
//  Created by Lucifer on 2022/8/1.
//

#include <functional>  // std::bind  // 参见 sg_slam/viewer.h #include <thread>
#include <unistd.h>  // usleep // 参见 sg_slam/viewer.cpp
#include <vector>
#include <iostream>
#include <string>
#include <torch/torch.h>

#include "sg_slam/loopclosing.h"
#include "sg_slam/config.h"
#include "sg_slam/g2o_types.h"
#include "sg_slam/algorithm.h"

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>  // cv::solvePnPRansac

#include "sg_slam/sg_detectMatcher.h"

#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>  // N.B.: 自定义优化类型使用吗??

namespace sg_slam {

// --------------------------------------------------------------------------------------------------------------
LoopClosing::LoopClosing() {
    m_loopclosing_running_.store(true);

    m_is_stop_flag.store(false);

    m_lcd_similarityScoreThreshold_high_ = Config::Get<float>(static_cast<std::string>("LoopClosing_lcd_similarityScoreThreshold_high"));
    m_lcd_similarityScoreThreshold_low_ = Config::Get<float>(static_cast<std::string>("LoopClosing_lcd_similarityScoreThreshold_low"));
    m_lcd_nDatabaseMinSize = static_cast<unsigned long>(Config::Get<int>(static_cast<std::string>("LoopClosing_nDatabaseMinSize")));

    // 参见 sg_slam/viewer.cpp Viewer::Viewer()
    m_show_LoopClosing_result_flag_ = ( Config::Get<int>(static_cast<std::string>("LoopClosing_show_LoopClosing_result_flag")) != 0 );

    // DeepLCD for loop detection
    m_deeplcd_ = DeepLCD::Ptr(new DeepLCD(-1));  // N.B.: 自改，与清华工程有所不同.

    // launch the loopclosing thread
    // 参见 sg_slam/viewer.h 
    m_loopclosing_thread_ = std::thread(std::bind(&LoopClosing::LoopClosingRun, this));

    m_insert_loopclosing_interval_ = static_cast<unsigned long>(Config::Get<int>(static_cast<std::string>("LoopClosing_insert_loopclosing_interval")));

    m_sg_loop_match_threshold_ = Config::Get<float>(static_cast<std::string>("sg_loop_match_threshold"));

    m_pose_graphloop_edge_num_ = static_cast<unsigned long>(Config::Get<int>(static_cast<std::string>("LoopClosing_pose_graphloop_edge_num")));

    m_pose_graph_most_old_loop_edge_num_ = static_cast<unsigned long>(Config::Get<int>(static_cast<std::string>("LoopClosing_pose_graph_most_old_loop_edge_num")));

    m_valid_feature_matches_th_ = Config::Get<int>(static_cast<std::string>("Frontend_num_features_Threshold_tracking"));

    m_valid_features_matches_ratio_th_ = Config::Get<double>(static_cast<std::string>("LoopClosing_ratio_features_th"));

    // 20221014加
    m_is_first_loopclosing_ = true;

    // 20221019
    m_detect_loop_number = 0;
    m_match_loop_number = 0;
    m_correct_number = 0;

    std::cout << std::endl;
    std::cout << "(LoopClosing::LoopClosing()): m_lcd_similarityScoreThreshold_high_ = " << m_lcd_similarityScoreThreshold_high_ << std::endl;
    std::cout << "(LoopClosing::LoopClosing()): m_lcd_similarityScoreThreshold_low_ = " << m_lcd_similarityScoreThreshold_low_ << std::endl;
    std::cout << "(LoopClosing::LoopClosing()): m_lcd_nDatabaseMinSize = " << m_lcd_nDatabaseMinSize << std::endl;
    std::cout << "(LoopClosing::LoopClosing()): m_show_LoopClosing_result_flag_ = " << m_show_LoopClosing_result_flag_ << std::endl;
    std::cout << "(LoopClosing::LoopClosing()): m_insert_loopclosing_interval_ = " << m_insert_loopclosing_interval_ << std::endl;
    std::cout << "(LoopClosing::LoopClosing()): m_sg_loop_match_threshold_ = " << m_sg_loop_match_threshold_ << std::endl;
    std::cout << "(LoopClosing::LoopClosing()): m_pose_graphloop_edge_num_ = " << m_pose_graphloop_edge_num_ << std::endl;
    std::cout << "(LoopClosing::LoopClosing()): m_pose_graph_most_old_loop_edge_num_ = " << m_pose_graph_most_old_loop_edge_num_ << std::endl;
    std::cout << "(LoopClosing::LoopClosing()): m_valid_feature_matches_th_ = " << m_valid_feature_matches_th_ << std::endl;
    std::cout << "(LoopClosing::LoopClosing()): m_valid_features_matches_ratio_th_ = " << m_valid_features_matches_ratio_th_ << std::endl;
    std::cout << "(LoopClosing::LoopClosing()): m_is_first_loopclosing_ = " << m_is_first_loopclosing_ << std::endl;
    std::cout << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
void LoopClosing::SetCameras(Camera::Ptr left, Camera::Ptr right) {
    m_cam_left_ = left;
    m_cam_right_ = right;
}

// --------------------------------------------------------------------------------------------------------------
void LoopClosing::SetMap(Map::Ptr map) {
    m_map_ = map;
}

// --------------------------------------------------------------------------------------------------------------
void LoopClosing::SetBackend(Backend::Ptr backend) {
    m_backend_ = backend;
}

// --------------------------------------------------------------------------------------------------------------
void LoopClosing::InsertProcessKeyframe() {
    std::unique_lock< std::mutex > lck(m_loopclosing_cache_mutex_);
    assert(m_map_ != nullptr);
    auto insertedKF = m_map_->GetCurrentKeyFrame();  // 前端刚插入地图的帧.
    // 5 (外化可调参数) KFs following the last closed KF will not be inserted
    // N.B.: 防止频繁检测，无用回环(间隔太近的回环， 作用不大)，应该在保证回环性能的情况下尽可能大，减少回环处理量，提高系统速度.
    // if((m_last_closed_KF == nullptr) || ((insertedKF->id_ - m_last_closed_KF->id_) > 5)) {
    // if((m_last_closed_KF_ == nullptr) || ((insertedKF->id_ - m_last_closed_KF_->id_) > m_insert_loopclosing_interval_)) {
    if((m_is_first_loopclosing_) || ((insertedKF->id_ - m_last_closed_KF_->id_) > m_insert_loopclosing_interval_)) {
        // unsigned long insertedKF_id = insertedKF->id_;
        // N.B.: 防止第一帧还没处理完就插入第二帧(太快了) 20221014
        m_is_first_loopclosing_ = false;
        m_process_keyframes_list_.push_back(insertedKF);
    } else {
        insertedKF->left_img_.release();
        insertedKF->left_cut_img_.release();
    }
}

// --------------------------------------------------------------------------------------------------------------
bool LoopClosing::CheckNeedProcessKeyframe() {
    std::unique_lock< std::mutex > lck(m_loopclosing_cache_mutex_);
    return (!m_process_keyframes_list_.empty());
}

// --------------------------------------------------------------------------------------------------------------
void LoopClosing::Stop() {
    // N.B.: 使用自己设计的等待缓存队列清空再停止线程的方法，不使用清华工程的方法.
    m_loopclosing_running_.store(false);
    m_loopclosing_thread_.join();
    std::cout << "Stop LoopClosing!" << std::endl;
    m_is_stop_flag.store(true);
}


// private function
// --------------------------------------------------------------------------------------------------------------
void LoopClosing::LoopClosingRun() {
    // N.B.: sg_slam/backend.cpp Backend::BackendLoop()
    while(m_loopclosing_running_.load() || CheckNeedProcessKeyframe()) {
        if(CheckNeedProcessKeyframe()) {
            // extract one KF to process from the list(缓冲区)
            // 参见 sg_slam/loopclosing.h ProcessNewKF()
            ProcessNewKF();

            // try to find the loop KF for the current KF
            // confirm: vt 确定/确认
            bool bConfirmedLoopKF = false;
            if(m_loopclosing_database_.size() > m_lcd_nDatabaseMinSize) {
                // 搜索回环
                if(DetectLoop()) {
                    m_detect_loop_number++;
                    // 对回环进行帧间匹配(为了计算修正量)
                    if(MatchFeatures()) {
                        m_match_loop_number++;
                        bConfirmedLoopKF = ComputeCorrectPose();
                        if(bConfirmedLoopKF) {
                            m_correct_number++;
                            LoopCorrect(); 
                        }
                    }
                }
            }

            // N.B.: 只有没有检测到回环帧的处理帧，可减少数据库的容量，
            // 保证有回环帧的帧，不会成为其他帧的回环帧, 不会形成一条长的回环帧依赖链路，控制搜索复杂度(不会沿着回环帧依赖链路搜索)
            // 使回环关系图仅有一级关系，但允许数据库的一个回环帧作为多个处理帧的回环帧.
            if(!bConfirmedLoopKF) {
                AddToDatabase();
            }
        }

        // 20221014 add
        // 当第一帧没有检测成功时，重新回退到等待第一帧的状态
        if(m_last_closed_KF_ == nullptr) {
            m_is_first_loopclosing_ = true;
        } else {
            m_is_first_loopclosing_ = false;
        }

        // N.B.: 没有使用信号阻塞, 个人认为可以不用 !!!!!
        // usleep(1000);  // 睡眠1000微秒，或者直到信号到达但未被阻塞或忽略。
    }

    auto pBackend = m_backend_.lock();
    // resume the backend
    pBackend->Resume();
    std::cout << "Stop LoopClosing::LoopClosingRun()!" << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
void LoopClosing::AddToDatabase() {
    // the KF(m_current_KF_) has been processed before (compute descriptors(deepLCD, superpoint) ... )

    // add current KF to the Database
    m_loopclosing_database_.insert( {m_current_KF_->id_, m_current_KF_} );
    m_last_KF_ = m_current_KF_;

    std::cout << std::endl;
    std::cout << "(LoopClosing::AddToDatabase()): add KF " << m_current_KF_->id_ << " to database. " << std::endl;
    std::cout << std::endl;

}

// --------------------------------------------------------------------------------------------------------------
void LoopClosing::ProcessNewKF() {
    {
        std::unique_lock< std::mutex > lck(m_loopclosing_cache_mutex_);
        m_current_KF_ = m_process_keyframes_list_.front();
        m_process_keyframes_list_.pop_front();  // 把队列第一个元素删除.

        // N.B.: 激活区不需要完全一致，若当前处理帧ID=3, (如果有的话)ID=4的帧也要调整，因为ID=4的帧的位姿时从ID=3推演过来的.
    }
    
    // calculate the whole image's descriptor vector with deeplcd.
    m_current_KF_->mpDeepDescrVector = m_deeplcd_->calcDescrOriginalImg(m_current_KF_->left_img_);

    // N.B.: 这里不计算 ORB 特征, 使用关键帧存储的深度特征及描述子进行帧间匹配即可.

    if(!m_show_LoopClosing_result_flag_) {
        // if doesn't neeed to show the match and reprojection result
        // then doesn't need to store the image.
        // N.B.: 若要显示也只有加入回环数据库中的关键帧回储存图片，检测到回环的关键帧处理结束后也会清除图片. !!!!!
        m_current_KF_->left_img_.release();
        m_current_KF_->left_cut_img_.release();
    }
}

// --------------------------------------------------------------------------------------------------------------
bool LoopClosing::DetectLoop() {
    // std::vector<float> vScores;
    float maxScore = 0;
    int cntSuspected = 0;  // suspected (不信任的).
    unsigned long bestId = 0;

    // 20221019 add
    /**
    if((m_last_closed_KF_ != nullptr) && ((m_current_KF_->id_ - m_last_closed_KF_->id_) > m_insert_loopclosing_interval_)) {
        return false;
    }
    **/   

    for(auto &db : m_loopclosing_database_) {
        // avoid comparing with recent KFs (避免与最近的 KF 比较)
        // 因为 m_loopclosing_database_ 默认对键(ID)进行升序排序，越前面的离得越远.
        // N.B.: 20 是判断最近的阈值，可以转为外化参数.
        if((m_current_KF_->id_ - db.first) < 20) {
            break;
        }

        float similarityScore = m_deeplcd_->score(m_current_KF_->mpDeepDescrVector, db.second->mpDeepDescrVector);
        if(similarityScore > maxScore) {
            // record the KF candidate with the max similarity score
            maxScore = similarityScore;
            bestId = db.first;
        }

        if(similarityScore > m_lcd_similarityScoreThreshold_low_) {
            cntSuspected++;
        }
    }

    // require high similarity score,
    // however, if there are too many high similarity scores, 
    // it means that current KF is not specific, then skip it.
    // N.B.: 3 可转为外化参数!!!!!
    // N.B.: 是候选回环帧的条件是与当前处理帧相似度最高且高于某一个阈值，
    // 并且当前处理帧与数据库中的其他(ID)不最近的关键帧的关键帧的相似度大都低于某一个阈值.
    if(maxScore < m_lcd_similarityScoreThreshold_high_ || cntSuspected > 3) {
        return false;
    }

    m_loop_KF_ = m_loopclosing_database_.at(bestId);

    std::cout << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "(LoopClosing::DetectLoop()): DeepLCD find potential Candidate KF " << m_loop_KF_->id_ << " for current KF " << m_current_KF_->id_ << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << std::endl;

    return true;
}

// --------------------------------------------------------------------------------------------------------------
bool LoopClosing::MatchFeatures() {
    
    // N.B.: 这里与清华的匹配方向相反，是从当前处理帧匹配到回环帧(因为有交叉验证，匹配方向不影响结果，只是为了逻辑统一)!!!!!
    auto desc_current = convertDesVector2Tensor(m_current_KF_->mDescriptors_data, m_current_KF_->Descripters_B, m_current_KF_->Descripters_D, m_current_KF_->Descripters_N);
    auto desc_loop = convertDesVector2Tensor(m_loop_KF_->mDescriptors_data, m_loop_KF_->Descripters_B, m_loop_KF_->Descripters_D, m_loop_KF_->Descripters_N);
    auto matcher_out_dict = sg_DetectMatcher::descMatch(desc_current, desc_loop, m_sg_loop_match_threshold_);
    auto matcher0to1_tensor = matcher_out_dict["matches0"][0].to(torch::kCPU);
    std::vector<cv::DMatch> matches_v;

    // N.B.: 没有改变原特征点集的大小.
    for(int64_t i=0; i < matcher0to1_tensor.size(0); i++) {
        int index0to1 = matcher0to1_tensor[i].item<int>();
        
        if(index0to1 > -1) {
            cv::DMatch match0to1_one;
            match0to1_one.queryIdx = i;
            match0to1_one.trainIdx = index0to1;
            matches_v.push_back(match0to1_one);
        }
    }

    auto RR_matchers_v = sg_RANSAC(m_current_KF_->mvKeys, m_loop_KF_->mvKeys, matches_v, 100);

    // the set(集合) to store valid matches (using std::pair<int, int> to represent the match);
    m_setValidFeatureMatches_.clear();

    for(auto &match : RR_matchers_v) {
        // https://www.javaroad.cn/questions/97155
        // N.B.: 与清华的不同采用自己设计的
        int currentFeatureIndex = match.queryIdx;
        int loopFeatureIndex = match.trainIdx;

        // the matches of keypoints belong to the same feature pair shouldn't be inserted into the valid matches twice.
        // 属于同一特征对的关键点的匹配不应两次插入到有效匹配中。
        if( m_setValidFeatureMatches_.find({currentFeatureIndex, loopFeatureIndex}) != m_setValidFeatureMatches_.end()) {
            continue;
        }

        // N.B.: 保证 m_setValidFeatureMatches_ 存储的对应特征，都不为空，因为操作空指针会引发段错误!!!!!!
        if(m_current_KF_->mvpfeatures[currentFeatureIndex] == nullptr) {
            continue;
        } 
        if(m_loop_KF_->mvpfeatures[loopFeatureIndex] == nullptr) {
            continue;
        }

        m_setValidFeatureMatches_.insert({currentFeatureIndex, loopFeatureIndex});
    }

    std::cout << std::endl;
    std::cout << "( LoopClosing::MatchFeatures()): number of valid feature matches: " << m_setValidFeatureMatches_.size() << std::endl;
    std::cout << std::endl;

    // N.B.: 10 可转化为外化参数，是回环帧间匹配是否成功的点数量阈值，
    // 由于使用的匹配器比ORB获得的优质匹配点较多, 故这个值可以设得大一点, 如50,100等，更好的避免引入错误的回环，
    // N.B.: 回环宁可不要，也不要引入错误的回环!!!!!
    // 20220922 改.

    size_t keys_number_min = m_current_KF_->mvKeysUn.size();
    if(m_loop_KF_->mvKeysUn.size() < keys_number_min) {
        keys_number_min = m_loop_KF_->mvKeysUn.size();
    }

    if(m_setValidFeatureMatches_.size() < static_cast<size_t>(m_valid_features_matches_ratio_th_ * keys_number_min)) {
        return false;
    }

    return true;
}

// --------------------------------------------------------------------------------------------------------------
// 先用opencv, 使用回环帧的3D地图点(world坐标下)和对应的当前处理帧的2D点(去畸变的)计算回环修正后的当前处理帧位姿(Tcw)的初始值，
// 再使用g2o 进行优化求解
bool LoopClosing::ComputeCorrectPose() {
    
    // prepare the data for PnP solver
    // 为 PnP 求解器准备数据
    // N.B.: 使用 cv::Point3d 会不会更好呢?
    // N.B.: 因为 cv::KeyPoint (关键点) 中的 pt 使用的是 cv::Point2f.
    std::vector< cv::Point3f > vLoopPoints3d;
    std::vector< cv::Point2f > vCurrentPoints2d;
    std::vector< cv::Point2f > vLoopPoints2d;
    std::vector< cv::DMatch > vMatchesWithMapPoint;

    // prepare the data for opencv solvePnPRansac()
    // remove the match whose loop feature is not linked to a mappoint.
    for(auto iter = m_setValidFeatureMatches_.begin(); iter != m_setValidFeatureMatches_.end(); ) {
        int currentFeatureIndex = (*iter).first;
        int loopFeatureIndex = (*iter).second;
        // auto mp = m_loop_KF_->mvpfeatures[loopFeatureIndex]->map_point_.lock();

        auto loop_KF_feat = m_loop_KF_->mvpfeatures[loopFeatureIndex];

        auto mp = m_loop_KF_->mvpfeatures[loopFeatureIndex]->GetMapPoint();

        // 参见 sg_slam/map.cpp void Map::CleanActiveMap()
        if(mp) {
            vCurrentPoints2d.push_back(m_current_KF_->mvpfeatures[currentFeatureIndex]->position_.pt);
            Vec3 pos = mp->GetPos();  // loop Mappoint
            vLoopPoints3d.push_back(cv::Point3f(pos(0), pos(1), pos(2)));
            vLoopPoints2d.push_back(m_loop_KF_->mvpfeatures[loopFeatureIndex]->position_.pt);

            // useful if needs to draw the matches
            cv::DMatch valid_match(currentFeatureIndex, loopFeatureIndex, 10.0);
            // cv::DMatch valid_match(currentFeatureIndex, loopFeatureIndex);  // 个人觉得用这个更好.
            vMatchesWithMapPoint.push_back(valid_match);

            iter++;
        } else {
            iter = m_setValidFeatureMatches_.erase(iter);
        }
    }

    std::cout << std::endl;
    std::cout << "(LoopClosing::ComputeCorrectPose()): number of valid matches with mappoints: " << vLoopPoints3d.size() << std::endl;
    std::cout << std::endl;

    if(m_show_LoopClosing_result_flag_) {
        // show the match result
        cv::Mat img_goodmatch;
        // 参见 sg_slam/KITTI_dataset.cpp KITTI_Dataset::NextFrame()
        if(m_current_KF_->is_use_cut_image) {
            cv::drawMatches(m_current_KF_->left_cut_img_, m_current_KF_->mvKeys, m_loop_KF_->left_cut_img_, m_loop_KF_->mvKeys, vMatchesWithMapPoint, img_goodmatch);
        } else {
            cv::drawMatches(m_current_KF_->left_img_, m_current_KF_->mvKeys, m_loop_KF_->left_img_, m_loop_KF_->mvKeys, vMatchesWithMapPoint, img_goodmatch);
        }
        
        cv::resize(img_goodmatch, img_goodmatch, cv::Size(), 0.5, 0.5);
        cv::imshow("(LoopClosing::ComputeCorrectPose()): valid matches with mappoints", img_goodmatch);
        cv::waitKey(1);
    }

    // N.B.: 10 可转化为外化参数，是回环帧间2D-3D匹配是否成功的点数量阈值，
    // 由于使用的匹配器比ORB获得的优质匹配点较多, 故这个值可以设得大一点, 如50,100等，更好的避免引入错误的回环，
    // N.B.: 回环宁可不要，也不要引入错误的回环!!!!!
    if(vLoopPoints3d.size() < 10) {
        return false;
    }

    // opencv: solve PnP with RANSAC (opencv计算的初始值)(用法与匹配的RANSAC不太一样，但原理一样)
    cv::Mat rvec, tvec, R, K;
    cv::eigen2cv(m_cam_left_->K(), K);
    Eigen::Matrix3d Reigen;
    // Mat33 Reigen;
    Eigen::Vector3d teigen;
    // Vec3 teigen;

    // N.B.: 使用“try-catch”，因为 cv::solvePnPRansac 可能会因为糟糕的匹配结果而失败，可能是因为 inlier 的重投影误差很高，导致的solvePnPRansac() 的结果不可靠.
    // use "try - catch" since cv::solvePnPRansac may fail because of terrible match result
    // and It may be that the result of solvePnPRansac() is unreliable due to the high reprojection error of inlier.
    // N.B.: https://blog.csdn.net/sss_369/article/details/92179738
    try {
        // 5.991 自由度为2 (3D-2D, 误差为2D)， 标准差为1个像素(已白化或假定测量特征点位置标准差为1个像素), 置信度为95%(alpha=0.05)的分位数(分位点)
        cv::solvePnPRansac(vLoopPoints3d, vCurrentPoints2d, K, cv::Mat(), rvec, tvec, false, 100, 5.991, 0.99);
    } catch(...) {
        return false;
    }

    cv::Rodrigues(rvec, R);  
    cv::cv2eigen(R, Reigen);  // Rcw
    cv::cv2eigen(tvec, teigen);  // tcw

    m_corrected_current_pose_ = Sophus::SE3d(Reigen, teigen);

    // use g2o optimization to further optimize the corrected current pose
    int cntInliers = OptimizeCurrentPose();

    std::cout << std::endl;
    std::cout << "(LoopClosing::ComputeCorrectPose()): number of match inliers (after optimization): " << cntInliers << std::endl;
    std::cout << std::endl;

    // N.B.: 10 可转化为外化参数，是优化后的回环修正后的当前处理帧的内点(重投影2d误差小于一定阈值)数量阈值，
    // 内点数过少，证明回环修正后的当前处理帧的位姿(Tcw)的估计效果不好, 故认为回环没有成功.
    // 由于使用的匹配器比ORB获得的优质匹配点较多, 即参与优化计算的2d-3d点对较多, 故这个值可以设得大一点, 如50,100等.
    // N.B.: 回环宁可不要，也不要引入错误的回环!!!!!
    if( cntInliers < 10 ) {
        return false;
    }

    /**
    // if the correct current pose is similar to current pose, 
    // then doesn't need to do LoopCorrectActive() and LoopCorrectPreviousKFandMappoint().
    // N.B.: 回环修正后的当前处理帧的位姿 与 未修正的当前处理帧的位姿很接近(很相似)，
    // 证明回环成功时，回环帧到当前帧这段连续帧的累积误差不大，故不需回环修正
    // N.B.: 度量位姿是否有大差异的方法, 1 可转化为外化参数, 用于判断位姿是否有大差异的阈值, 若有大差异，才进行回环修正.
    double error = (m_current_KF_->Pose() * m_corrected_current_pose_.inverse()).log().norm();
    if(error > 1.0) {
        m_bNeedCorrect_flag = true;
    } else {
        m_bNeedCorrect_flag = false;
    }
    **/
    // m_bNeedCorrect_flag = true;

    /**
     * N.B.: 这里可加入回环帧与当前处理帧的地图点的融合，无论是否进行回环修正，只要回环成功，就可以融合地图点。
     */

    // show the reprojection result (将回环帧的地图点投影到当前处理帧的图像坐标上(去畸变的))
    if(m_show_LoopClosing_result_flag_) {
        Vec3 t_eigen = m_corrected_current_pose_.translation();
        Mat33 R_eigen = m_corrected_current_pose_.rotationMatrix();
        cv::Mat R_cv, t_cv, r_cv;
        cv::eigen2cv(R_eigen, R_cv);
        cv::eigen2cv(t_eigen, t_cv);
        cv::Rodrigues(R_cv, r_cv);
        std::vector<cv::Point2f> vReprojectionPoints2d;
        std::vector<cv::Point2f> vCurrentKeyPoints;  // N.B.: 统一使用去畸变的关键点更好
        std::vector<cv::Point3f> vLoopMapPoints;

        // N.B.: m_setValidFeatureMatches_ 已经去掉了回环帧特征没有对应地图点的匹配
        for(auto iter = m_setValidFeatureMatches_.begin(); iter != m_setValidFeatureMatches_.end(); iter++) {
            int currentFeatureIndex = (*iter).first;
            int loopFeatureIndex = (*iter).second;
            // auto mp = m_loop_KF_->mvpfeatures[loopFeatureIndex]->map_point_.lock();
            auto mp = m_loop_KF_->mvpfeatures[loopFeatureIndex]->GetMapPoint();
            vCurrentKeyPoints.push_back(m_current_KF_->mvpfeatures[currentFeatureIndex]->position_.pt);
            Vec3 pos = mp->GetPos();
            vLoopMapPoints.push_back(cv::Point3f(pos(0), pos(1), pos(2)));
        }
        // do the reprojection
        cv::projectPoints(vLoopMapPoints, r_cv, t_cv, K, cv::Mat(), vReprojectionPoints2d);

        // show the reprojection result
        cv::Mat imgOut;
        if(m_current_KF_->is_use_cut_image) {
            cv::cvtColor(m_current_KF_->left_cut_img_, imgOut, cv::COLOR_GRAY2RGB);
        } else {
            cv::cvtColor(m_current_KF_->left_img_, imgOut, cv::COLOR_GRAY2RGB);
        }
        
        for(size_t index = 0; index < vLoopMapPoints.size(); index++) {
            cv::circle(imgOut, vCurrentKeyPoints[index], 5, cv::Scalar(0, 0, 255), -1);
            cv::line(imgOut, vCurrentKeyPoints[index], vReprojectionPoints2d[index], cv::Scalar(255, 0, 0), 2);
        }

        cv::imshow("(LoopClosing::ComputeCorrectPose()): reprojection result of match inliers", imgOut);
        cv::waitKey(1);
    }

    // 现在已通过所有验证，将其视为真正的循环关键帧
    // now has passed all verification, regard it as the true loop keyframe
    // N.B.: 注: 当前处理帧的位姿(Tcw)还没有发生变化，还不是回环修正后的的位姿.
    m_current_KF_->mpLoopKF = m_loop_KF_;
    m_current_KF_->isLoop = true;  
    m_current_KF_->m_relative_pose_to_loopKF = m_corrected_current_pose_ * m_loop_KF_->Pose().inverse();  // Tcr (回环修正后的).
    // m_last_closed_KF_ = m_current_KF_;  // N.B.: 将此句改在 LoopClosing::LoopCorrect() 最后， 因为需要在 PoseGraphOptimization() 使用上一个回环帧的信息.

    return true;
}

// --------------------------------------------------------------------------------------------------------------
// 参见 sg_slam/backend.cpp void Backend::Optimize
// 只优化一个节点.
int LoopClosing::OptimizeCurrentPose() {
    // 构建图优化，先设定g2o
    // setup g2o
    // solver for PnP/3D VO
    // pose dimension is 6 (se3), landmark dimension is 3.
    using BlockSolverType = g2o::BlockSolver_6_3;
    // 线性求解器类型(稠密)
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
    
    // use LM (N.B.: 可以换优化搜索算法)
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;  // N.B.: 图模型，不论稀疏还是稠密都是用这个求解器类型.
    optimizer.setAlgorithm( solver );  // 设置求解器.

    // vertex
    VertexPose *vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(m_corrected_current_pose_);
    optimizer.addVertex(vertex_pose);

    // K ((左)内参)
    Mat33 K = m_cam_left_->K();

    // edges, id从1开始
    int index = 1;
    // 一元边
    std::vector< EdgeProjectionPoseOnly* > edges;
    // m_setValidFeatureMatches_已经移除了回环帧特征没有对应地图点的匹配对
    // N.B.: https://blog.csdn.net/sinat_38183777/article/details/82056289
    edges.reserve(m_setValidFeatureMatches_.size());
    std::vector<bool> vEdgeIsOutlier;
    vEdgeIsOutlier.reserve(m_setValidFeatureMatches_.size());
    std::vector< std::pair<int, int> > vMatches;
    vMatches.reserve(m_setValidFeatureMatches_.size());

    // N.B.: 可转换为外化参数.
    const double chi2_th = 5.991;  // robust kernel 阈值 (可调)
    for(auto iter = m_setValidFeatureMatches_.begin(); iter != m_setValidFeatureMatches_.end(); iter++) {
        int currentFeatureIndex = (*iter).first;
        int loopFeatureIndex = (*iter).second;
        // auto mp = m_loop_KF_->mvpfeatures[loopFeatureIndex]->map_point_.lock();
        auto mp = m_loop_KF_->mvpfeatures[loopFeatureIndex]->GetMapPoint();
        auto point2d = m_current_KF_->mvpfeatures[currentFeatureIndex]->position_.pt;
        // m_setValidFeatureMatches_已经移除了回环帧特征没有对应地图点的匹配对
        assert(mp != nullptr);  // N.B.: 经过上面操作，这个断语都会通过

        EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(mp->GetPos(), K);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(toVec2(point2d));
        // 信息矩阵默认设为单位矩阵, 故测量误差最好白化，或者根据方差设置信息矩阵.
        edge->setInformation(Eigen::Matrix2d::Identity());
        auto rk = new g2o::RobustKernelHuber();
        // N.B.: delta 默认为 1.
        rk->setDelta(chi2_th);  // N.B.: 自己加的，回环效果不好再去掉!!!!! (清华工程没有)
        // std::cout << "(LoopClosing::OptimizeCurrentPose()): delta: " << std::endl;
        edge->setRobustKernel(rk);
        edges.push_back(edge);
        vEdgeIsOutlier.push_back(false);
        vMatches.push_back(*iter);

        optimizer.addEdge(edge);

        index++;
    }

    // estimate the Pose and determine the outliers
    // start optimization
    int cntOutliers = 0;
    // N.B.: 可转换为外化参数. 迭代次数(可调)
    int numIterations = 4;

    // N.B.: 这里的优化，沿用与十四讲前端相同逻辑，与清华工程的逻辑不同，可实验验证哪个逻辑精度更高.
    // optimizer.initializeOptimization();
    // optimizer.optimize(10); // N.B.: 10 可转换为外化参数. 参见 sg_slam/backend.cpp Backend::Optimize

    // use the same strategy as in frontend;
    for(int iteration=0; iteration < numIterations; iteration++) {
        vertex_pose->setEstimate(m_corrected_current_pose_);  // 重新设置初值 // N.B.: 十四讲有，清华工程没有，个认为十四讲考虑的更好，剔除外点后应该回到同一起点开始迭代 !!!!!
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cntOutliers = 0;

        // count the outliers
        for(size_t i=0, N = edges.size(); i < N;  i++) {
            auto e = edges[i];
            if(vEdgeIsOutlier[i]) {
                // 上一次是外点的，因为这一次迭代没有优化，需要在这一次迭代的基础上重新计算误差，在判断是否还是外点.
                e->computeError(); 
            }

            if(e->chi2() > chi2_th) {
                vEdgeIsOutlier[i] = true;
                e->setLevel(1);  // 不优化
                cntOutliers++;  
            } else {
                vEdgeIsOutlier[i] = false;
                e->setLevel(0);  // 优化
            }

            if(iteration == numIterations - 2) {
                e->setRobustKernel(nullptr);  // 因为剔除了错误的边，所以倒数第二次之后优化不再使用核函数（(即最后一次优化不使用核函数)
            }
        }
    }

    // remove the outlier match
    for(size_t i = 0, N = vEdgeIsOutlier.size(); i<N; i++) {
        if(vEdgeIsOutlier[i]) {
            m_setValidFeatureMatches_.erase(vMatches[i]);
        }
    }

    m_corrected_current_pose_ = vertex_pose->estimate();

    return static_cast<int>(m_setValidFeatureMatches_.size());
}

// --------------------------------------------------------------------------------------------------------------
void LoopClosing::LoopCorrect() {

    /**
    if(!m_bNeedCorrect_flag) {
        std::cout << std::endl;
        std::cout << "(LoopClosing::LoopCorrect()): LoopClosing: no need for correction. " << std::endl;
        std::cout << std::endl;
        return;
    }
    **/

    // request the backend to pause, avoiding conflict.
    // 请求后端暂停，避免冲突。
    // lock()如果weak_ptr对象已过期，返回一个空shared_ptr对象；否则，返回一个weak_ptr内部指针相关联的shared_ptr对象（因此，shared_ptr对象的引用计数+1）
    auto pBackend = m_backend_.lock();
    pBackend->RequestPause();
    while( !pBackend->IfHasPaused() ) {
        // 参见 sg_slam/backend.cpp  Backend::BackendLoop()
        // N.B.: 清华工程有加，这里个人认为去掉更好，实验验证.
        // usleep(1000);  
    }

    std::cout << "(LoopClosing::LoopCorrect()): Backend has paused." << std::endl;

    // correct the KFs and mappoints in the active map
    // N.B.: 功能修改只进行地图点融合不进行地图点和关键帧修正，修正统一在PoseGraphOptimization中执行.
    LoopCorrectActive();

    // optimize all the previous KFs' poses using pose graph optimization
    PoseGraphOptimization();

    // resume the backend
    pBackend->Resume();


    if( m_last_loopKPs_id_list_.size() >= m_pose_graphloop_edge_num_) {

        if(m_most_old_loopKPS_id_list_.size() < m_pose_graph_most_old_loop_edge_num_) {
            auto old_id = m_last_loopKPs_id_list_.front();
            m_most_old_loopKPS_id_list_.push_back(old_id);
        }

        m_last_loopKPs_id_list_.pop_front();
    }

    m_last_loopKPs_id_list_.push_back(m_current_KF_->id_);

    m_last_closed_KF_ = m_current_KF_;

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "(LoopClosing::LoopCorrect()): LoopClosing: Correction done." << std::endl;
    std::cout << "m_last_loopKPs_id_list_.size(): " << m_last_loopKPs_id_list_.size() << std::endl;
    std::cout << "m_most_old_loopKPS_id_list_.size(): " << m_most_old_loopKPS_id_list_.size() << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

}

// --------------------------------------------------------------------------------------------------------------
void LoopClosing::LoopCorrectActive() {
    
    
    // avoid the conflict between frontend tracking and loopclosing correction.
    // N.B.: 上述是清华工程在此加锁的原因，但在本系统中由于前端只估计普通帧相对于参考关键帧的位姿，
    // 且仅生成关键帧插入地图，后端，回环时才计算并更新关键帧的数据，一定在回环和后端处理之前，故不会与回环产生冲突，
    // 则不需要加锁

    std::unordered_map<unsigned long, SE3> correctedActivePoses;
    // N.B.: 这里与清华工程有所不同
    correctedActivePoses.insert( {m_current_KF_->id_, m_corrected_current_pose_} );

    // calculate the relative pose between current KF and KFs in active map
    // and insert corrected pose of KFs to the correctedActivePoses (unordered)map
    // and correct the mappoints' positions.
    auto active_keyframes_map = m_map_->GetActiveKeyFrames();
    for(auto &keyframe : active_keyframes_map) {
        unsigned long kfId = keyframe.first;
        // N.B.: 不把当前帧添加到映射集合(不改变原本的对应位姿(回环修正后的位姿)) 
        if(kfId == m_current_KF_->id_) {
            continue;
        }

        // 用回环修正前的当前处理帧位姿，计算相对位姿
        auto keyframe_a = keyframe.second;
        // SE3 Tac = keyframe.second->Pose() * (m_current_KF_->Pose().inverse());
        SE3 Tac = keyframe_a->Pose() * (m_current_KF_->Pose().inverse());
        // 用回环修正后的当前处理帧位姿与相对位姿，计算修正位姿初值(Taw)
        SE3 Ta_corrected = Tac * m_corrected_current_pose_;

        correctedActivePoses.insert({kfId, Ta_corrected});
    }

    // correct the active mappoints 
    m_setCorrectedMappoints_id_.clear();
    auto activeMappoints = m_map_->GetActiveMapPoints();
    for(auto &mappoint : activeMappoints) {
        MapPoint::Ptr mp = mappoint.second;
        if(mp->GetActiveObs().empty()) {
            continue;
        }

        auto refKeyId = mp->reference_KeyFrame_.lock()->id_;
        // N.B.: 不是所有激活帧都调整
        if(correctedActivePoses.find(refKeyId) == correctedActivePoses.end()) {
            continue;
        }

        auto refKey = mp->reference_KeyFrame_.lock();
        Vec3 posCamera = refKey->Pose() * mp->GetPos();

        SE3 Ta_corrected = correctedActivePoses.at(refKeyId);
        mp->SetPos(Ta_corrected.inverse() * posCamera);
        if(m_setCorrectedMappoints_id_.find(mp->id_) == m_setCorrectedMappoints_id_.end()) {
            m_setCorrectedMappoints_id_.insert(mp->id_);
        }
    }

    m_setCorrectedKeyframes_id_.clear();
    // then correct the active KFs' poses
    for(auto &keyframe : active_keyframes_map) {
        auto kfId = keyframe.first;
        if(correctedActivePoses.find(kfId) == correctedActivePoses.end()) {
            continue;
        }
        keyframe.second->SetPose(correctedActivePoses.at(kfId));
        if(m_setCorrectedKeyframes_id_.find(kfId) == m_setCorrectedKeyframes_id_.end()) {
            m_setCorrectedKeyframes_id_.insert(kfId);
        }
    }

    // replace the current KF's mappoints with loop KF's matched mappoints
    // N.B.: 用循环 KF 的匹配映射点替换当前 KF 的映射点. (旧点换新点) 
    // (不把旧点加入当前激活序列，只是把被替代点移出，下一次前端帧间如果匹配上会增加激活观测，故融合时不用管激活观测，最多是不会立刻参与优化时)
    // N.B.: 进行修正后才进行, 回环帧与当前处理帧的地图点的融合(N.B.: 个人持不同意见).
    // N.B.: 个人认为: 无论是否进行回环修正，只要回环成功，就可以融合地图点。(这句话可作为搜索索引)
    // N.B.: 可以实验验证那个处理流程好
    for(auto iter = m_setValidFeatureMatches_.begin(); iter != m_setValidFeatureMatches_.end(); iter++) {
        
        int currentFeatureIndex = (*iter).first;
        int loopFeatureIndex = (*iter).second;

        // auto loop_mp = m_loop_KF_->mvpfeatures[loopFeatureIndex]->map_point_.lock();
        auto loop_mp = m_loop_KF_->mvpfeatures[loopFeatureIndex]->GetMapPoint();
        // m_setValidFeatureMatches_已经移除了回环帧特征没有对应地图点的匹配对
        assert(loop_mp != nullptr);

        if(loop_mp) {
            // auto current_mp = m_current_KF_->mvpfeatures[currentFeatureIndex]->map_point_.lock();
            auto current_mp = m_current_KF_->mvpfeatures[currentFeatureIndex]->GetMapPoint();
            // 
            if(current_mp) {
                // link the current mappoint's observation to the matched loop mappoint
                for(auto &obs : current_mp->GetObs()) {
                    auto obs_feat = obs.lock();
                    loop_mp->AddObservation(obs_feat);
                    // obs_feat->map_point_ = loop_mp;
                    obs_feat->SetMapPoint(loop_mp);
                }

                // then, remove the current mappoint from the map(全局和激活)
                m_map_->RemoveMapPoint(current_mp);
            } else {
                // m_current_KF_->mvpfeatures[currentFeatureIndex]->map_point_ = loop_mp;
                m_current_KF_->mvpfeatures[currentFeatureIndex]->SetMapPoint(loop_mp);
            }
        }
    }
}

// -------------------------------------------------------------------------------------------
void LoopClosing::PoseGraphOptimization() {
    using BlockSolverType = g2o::BlockSolver< g2o::BlockSolverTraits<6, 6> >;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
    auto solver = new g2o::OptimizationAlgorithmLevenberg( 
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()) );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    auto allKFs = m_map_->GetAllKeyFrames();

    // vertices
    std::map<unsigned long, VertexPose*> vertices_kf;
    for(auto &keyframe : allKFs) {
        unsigned long kfId = keyframe.first;
        KeyFrame::Ptr kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();
        vertex_pose->setId(kf->id_);
        vertex_pose->setEstimate(kf->Pose());
        vertex_pose->setMarginalized(false);

        // 设置不优化修正的节点，仅作为约束
        // auto mapActiveKFs = m_map_->GetActiveKeyFrames();
        
        // 20221012 改
        /**
        if( (m_setCorrectedKeyframes_id_.find(kfId) != m_setCorrectedKeyframes_id_.end()) 
                || (kfId == m_loop_KF_->id_) || (kfId == 0) ) {
            vertex_pose->setFixed(true);
        }
        **/
        if( (m_setCorrectedKeyframes_id_.find(kfId) != m_setCorrectedKeyframes_id_.end()) 
                || (kfId == m_loop_KF_->id_) ) {
            vertex_pose->setFixed(true);
        }
        
        /**
        if( (kfId == m_loop_KF_->id_) || (kfId == 0) ) {
            vertex_pose->setFixed(true);
        }
        **/
        
        /**
        if( (kfId == m_current_KF_->id_) || (kfId == m_loop_KF_->id_) || (kfId == 0) ) {
            vertex_pose->setFixed(true);
        }
        **/
        

        optimizer.addVertex(vertex_pose);
        vertices_kf.insert( {kf->id_, vertex_pose} );
    }

    // edges
    // int index = 0;  // N.B.: 原工程从0开始
    int index = 1;  // N.B.: 自己改为从1开始
    // std::map<int, EdgePoseGraph*> vEdges;
    for(auto &keyframe : allKFs) {
        unsigned long kfId = keyframe.first;
        assert(vertices_kf.find(kfId) != vertices_kf.end());
        auto kf = keyframe.second;

        // edge type 1: edge between two KFs adjacent in time;
        // 时间上相邻的两个 KF 之间的边
        // (个人设定的)约束1: 认为关键帧与上一帧的相对位姿应该不变. (除了第一帧(ID==0), 其他关键帧均有这个约束)
        // N.B.: 此约束保证修正后的轨迹不会有明显断裂.
        auto lastKF = kf->mpLastKF.lock();
        if(lastKF) {
            EdgePoseGraph *edge = new EdgePoseGraph();
            edge->setId(index);
            edge->setVertex(0, vertices_kf.at(kfId));
            edge->setVertex(1, vertices_kf.at(lastKF->id_));
            edge->setMeasurement(kf->m_relative_pose_to_lastKF);
            edge->setInformation(Mat66::Identity());
            optimizer.addEdge(edge);
            // vEdges.insert({index, edge});
            index++;
        }
        

        auto nextKF = kf->mpNextKF.lock();
        if(nextKF) {
            EdgePoseGraph *edge = new EdgePoseGraph();
            edge->setId(index);
            edge->setVertex(0, vertices_kf.at(kfId));
            edge->setVertex(1, vertices_kf.at(nextKF->id_));
            edge->setMeasurement(kf->m_relative_pose_to_nextKF);
            edge->setInformation(Mat66::Identity());
            optimizer.addEdge(edge);
            index++;
        }

        // edge type 2: loop edge
        /**
        // (个人设定)约束2: 认为已经有回环帧的关键帧(包括当前除处理帧), 与对应回环帧的相对位姿应该不变
        // N.B.: 此约束最大限度地保留了回环的修正信息.
        auto loopKF = kf->mpLoopKF.lock();

        if(loopKF) {
            EdgePoseGraph *edge = new EdgePoseGraph();
            edge->setId(index);
            edge->setVertex(0, vertices_kf.at(kfId));
            edge->setVertex(1, vertices_kf.at(loopKF->id_));
            edge->setMeasurement(kf->m_relative_pose_to_loopKF);
            edge->setInformation(Mat66::Identity());
            optimizer.addEdge(edge);
            // vEdges.insert({index, edge});
            index++;
        }
        **/
        
        // N.B.: 此关键帧是上几次(最近几次)检测到回环的关键帧，就引入回环边约束
        // N.B.: 自加
        for(auto it=m_last_loopKPs_id_list_.begin(); it!=m_last_loopKPs_id_list_.end(); it++) {
            // N.B.: 本系统m_last_loopKPs_id_list_不会有重复元素
            if(kf->id_ == *it) {
                auto loopKF = kf->mpLoopKF.lock();

                if(loopKF) {
                    EdgePoseGraph *edge = new EdgePoseGraph();
                    edge->setId(index);
                    edge->setVertex(0, vertices_kf.at(kfId));
                    edge->setVertex(1, vertices_kf.at(loopKF->id_));
                    edge->setMeasurement(kf->m_relative_pose_to_loopKF);
                    edge->setInformation(Mat66::Identity());
                    optimizer.addEdge(edge);
                    index++;
                    break;
                }
            }
        }

        // N.B.: 自加
        for(auto it=m_most_old_loopKPS_id_list_.begin(); it!=m_most_old_loopKPS_id_list_.end(); it++) {
            // N.B.: 本系统m_most_old_loopKPS_id_list_不会有重复元素,
            // N.B.: m_most_old_loopKPS_id_list_ 
            if(kf->id_ == *it) {
                auto loopKF = kf->mpLoopKF.lock();

                if(loopKF) {
                    EdgePoseGraph *edge = new EdgePoseGraph();
                    edge->setId(index);
                    edge->setVertex(0, vertices_kf.at(kfId));
                    edge->setVertex(1, vertices_kf.at(loopKF->id_));
                    edge->setMeasurement(kf->m_relative_pose_to_loopKF);
                    edge->setInformation(Mat66::Identity());
                    optimizer.addEdge(edge);
                    index++;
                    break;
                }
            }
        }
    }

    // do the optimization
    // N.B.: 量大仅优化一次即可
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    {
        // set the mappoints' position according to its reference keyframe's optimized pose.
        auto allMapPoints = m_map_->GetAllMapPoints();
        // auto activeMapPoints = m_map_->GetActiveMapPoints();
        for(auto &mappoint : allMapPoints) {
            auto mp_id = mappoint.first;
            auto mp = mappoint.second;
            // N.B.: 后端可能清除全局地图点的观测，但全局地图还没来得及清除全局观测为0的点导致这个断语触发.
            // assert(!mp->GetObs().empty());
            if(mp->GetObs().empty()) {
                continue;
            }

            // N.B.: 与清华工程的不同，上面激活区地图点的修正是使用以激活关键帧找激活点的方式
            // N.B.: 激活区的地图点已经修正过，跳过
            auto refKeyId = mp->reference_KeyFrame_.lock()->id_;
            if(m_setCorrectedMappoints_id_.find(mp_id) != m_setCorrectedMappoints_id_.end()) {
                continue;
            }

            // N.B.: 个人认为，这一个判断有没有没有影响!!!!!
            if(vertices_kf.find(refKeyId) == vertices_kf.end()) {
                // NOTICE: this is for the case that one mappoint is inserted into map in frontend thread
                // but the KF which first observes it hasn't been inserted into map in backend thread
                continue;
            }

            auto refKey = mp->reference_KeyFrame_.lock();
            Vec3 posCamera = refKey->Pose() * mp->GetPos();

            SE3 T_optimized = vertices_kf.at(refKeyId)->estimate();
            mp->SetPos(T_optimized.inverse() * posCamera);
        }

        // set the KFs' optimized poses (包括当前处理帧和对应的回环帧)
        for(auto &v : vertices_kf) {
            allKFs.at(v.first)->SetPose(v.second->estimate());
        }
    }

    std::cout << std::endl;
    std::cout << "(LoopClosing::PoseGraphOptimization()): Pose graph optimization done." << std::endl;
    std::cout << std::endl;
}

}  // namespace sg_slam
