//
//  Created by Lucifer on 2022/7/25.
//

#include <functional>  // std::bind  // 参见 sg_slam/viewer.h #include <thread>
#include <unistd.h>  // usleep

#include "sg_slam/viewer.h"
#include "sg_slam/config.h"
#include "sg_slam/common_include.h"  

namespace sg_slam {

float Viewer::Viewer_sz = 1.0f;  // N.B.: 这样实现不好(但改动量)较小!!!! 20221028

// --------------------------------------------------------------------------------------------------------------
Viewer::Viewer() {
    m_is_stop_flag.store(false);

    delay_mT_ = Config::Get<int>(static_cast<std::string>("Viewer_cv_wait_mT"));
    // N.B.: yaml 无法解析 bool.
    m_viewer_running_ = ( Config::Get<int>(static_cast<std::string>("Viewer_running_flag")) != 0 );
    if(delay_mT_ < 0) {
        delay_mT_ = 0;
    }
    // 2022101101 加
    int cut_flag = Config::Get<int>(static_cast<std::string>("camera.cut"));
    if( cut_flag == 0 ) {
        m_image_width_ = Config::Get<int>(static_cast<std::string>("camera.width"));
        m_image_height_ = Config::Get<int>(static_cast<std::string>("camera.height"));
    } else {
        m_image_width_ = Config::Get<int>(static_cast<std::string>("camera.cut_width"));
        m_image_height_ = Config::Get<int>(static_cast<std::string>("camera.cut_height"));
    }

    viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));
}

// --------------------------------------------------------------------------------------------------------------
void Viewer::Close() {
    m_viewer_running_ = false;
    viewer_thread_.join();
    std::cout << "Stop viewer!" << std::endl;
    m_is_stop_flag.store(true);
}

// --------------------------------------------------------------------------------------------------------------
void Viewer::AddCurrentFrame(Frame::Ptr currentFrame) {
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);  // N.B.: 前端插入需要加锁.
    current_frame_ = currentFrame;
}

// --------------------------------------------------------------------------------------------------------------
// 自己改 仅更新 frame_results_map_和所有地图点
// 一直在清屏幕，需要一直画。 故 map_updated_ 标志无用
void Viewer::UpdateMap() {
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);  // 前端插入点和关键帧及帧结果更新地图结构.
    assert(map_ != nullptr);
    // map_updated_ = false;  // My add (N.B.: 若显示不好可去掉)
    frame_results_map_ = map_->GetAllFrameResults();
    landmarks_map_ = map_->GetAllMapPoints();
    // map_updated_ = true;
}


// private function
// --------------------------------------------------------------------------------------------------------------
void Viewer::ThreadLoop() {
    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Viewer::ThreadLoop is working ..." << std::endl;

    const int UI_WIDTH = 175;
    int WINDOW_W = 1024+UI_WIDTH;
    pangolin::CreateWindowAndBind("SG_SLAM", WINDOW_W, 768);
    glEnable(GL_DEPTH_TEST);  // 用来开启更新深度缓冲区的功能
    glEnable(GL_BLEND); // 启用混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // 混合函数

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
    pangolin::Var<bool> menuShowFrames("menu.Show Frames", true, true);
    pangolin::Var<bool> menuSaveImg("menu.Save Img", false, false);
    pangolin::Var<bool> menuSaveWin("menu.Save Win", false, false);

    float viewpointX = Config::Get<float>(static_cast<std::string>("Viewer.ViewpointX"));
    float viewpointY = Config::Get<float>(static_cast<std::string>("Viewer.ViewpointY"));
    float viewpointZ = Config::Get<float>(static_cast<std::string>("Viewer.ViewpointZ"));
    float viewpointF = Config::Get<float>(static_cast<std::string>("Viewer.ViewpointF"));

    // 创建一个观察相机
    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, viewpointF, viewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View &vis_display = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(vis_camera));

    bool bFollow = true;

    // 20230308 加
    pangolin::View &current_img_display = pangolin::CreateDisplay()
        .SetBounds(7.0/9.0f, 1.0f, pangolin::Attach::Pix(UI_WIDTH), 4.0/9.0f, -1024.0f / 768.0f)
        .SetLock(pangolin::LockLeft, pangolin::LockTop);
    pangolin::GlTexture current_imgTexture(m_image_width_, m_image_height_, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);

    while (!pangolin::ShouldQuit() && m_viewer_running_)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // 背景为黑 // 一直在清屏幕，需要一直画。
        glClearColor(0.5f, 0.5f, 0.5f, 0.5f);  // 背景为黑 // 一直在清屏幕，需要一直画。
        // glClearColor(1.0f, 1.0f, 1.0f, 0.5f);  // 背景为白(透明) // 一直在清屏幕，需要一直画。
        vis_display.Activate(vis_camera);

        std::unique_lock<std::mutex> lck(viewer_data_mutex_);
        // current_frame_不为空指针
        if(current_frame_) {
            if(menuFollowCamera && bFollow) {
                // 持续跟踪当前帧
                FollowCurrentFrame(vis_camera);
            } else if(!menuFollowCamera && bFollow) {
                // 不跟踪当前帧
                bFollow = false;
                // 无 FollowCurrentFrame(vis_camera); 就没有跟踪当前帧视角.
            } else if(menuFollowCamera && !bFollow) {
                // 重新跟踪当前帧时需要对齐当前帧的位置.
                // N.B.: 自己调整了顺序，如果显示不好再换回来!!!!!
                vis_camera.SetModelViewMatrix(
                    pangolin::ModelViewLookAt(viewpointX,viewpointY,viewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
                );
                FollowCurrentFrame(vis_camera);
                bFollow = true;
            }
        }

        // if(current_frame_) 中前面会调整 vis_camera 的数据
        
        vis_display.Activate(vis_camera); 

        // 地图指针不为空
        if(current_frame_) {
            DrawFrame(current_frame_, m_purple_);
        }
        if(map_) {
            DrawMapPointsAndFrameResult(menuShowFrames, menuShowPoints);
        }

        if(current_frame_) {
            
            // DrawFrame(current_frame_, m_purple_);
            cv::Mat img = PlotFrameImage();
            // 向GPU装载图像
            // 20230308 加
            current_imgTexture.Upload(img.data, GL_RGB, GL_UNSIGNED_BYTE);
            current_img_display.Activate();
            glColor3f(1.0f, 1.0f, 1.0f); // 设置默认背景色，对于显示图片来说，不设置也没关系
            current_imgTexture.RenderToViewportFlipY(); // 需要反转Y轴，否则输出是倒着的

            // cv::imshow("current_frame", img);
            // cv::waitKey(delay_mT_);  // delay xx ms
        }

        

        // 检测按键是否按下
        if( pangolin::Pushed(menuSaveImg) ) {
            // vis_display.SaveOnRender("result");
            pangolin::SaveWindowOnRender("result", vis_display.GetBounds());
        }

        if( pangolin::Pushed(menuSaveWin) ) {
            pangolin::SaveWindowOnRender("result_window");
        }

        pangolin::FinishFrame();
        // usleep(5000);  // 睡眠5000微秒，或者直到信号到达但未被阻塞或忽略。
        usleep(1000);
    }

    // N.B.: 不加这一句, 会使main最后返回 return 0 有段错误.
    pangolin::DestroyWindow("SG_SLAM");
    
    std::cout << "Stop Viewer::ThreadLoop()!" << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
cv::Mat Viewer::PlotFrameImage() {
    cv::Mat img_out;
    // 默认输入图像是灰度图.
    if(current_frame_->left_img_.type() == CV_8UC1) {
        if(current_frame_->is_use_cut_image) {
            cv::cvtColor(current_frame_->left_cut_img_, img_out, CV_GRAY2BGR);
        } else {
            cv::cvtColor(current_frame_->left_img_, img_out, CV_GRAY2BGR);
        }
        
    } else {
        // img_out = current_frame_->left_img_;
        if(current_frame_->is_use_cut_image) {
            img_out = current_frame_->left_cut_img_.clone();
        } else {
            img_out = current_frame_->left_img_.clone();
        }
    }

    for(size_t i=0; i < current_frame_->mvpMapPoints.size(); i++) {
        // N.B.: 这个判定需要，有可能有关键点没有对应的地图点故，地图点为空(nullptr)
        if(current_frame_->mvpMapPoints[i].lock()) {
            cv::circle(img_out, current_frame_->mvKeys[i].pt, 2, cv::Scalar(0, 255, 0), 2);
        }
    }

    return img_out;
}

// --------------------------------------------------------------------------------------------------------------
void Viewer::DrawFrame(Frame::Ptr frame, const float *color) {
    SE3 Tcr = frame->RelativePose();
    SE3 Trw = frame->reference_keyframe_ptr_.lock()->Pose();
    SE3 Twc = (Tcr * Trw).inverse();

    // const float sz = 1.0f;
    const float sz = Viewer_sz;
    // const float sz = 0.10f;
    //const int line_width = 2;
    const float line_width = 2.0f;
    const float fx = 400.0f;
    const float fy = 400.0f;
    const float cx = 512.0f;
    const float cy = 384.0f;
    const float width = 1080.0f;
    const float height = 760.0f;

    glPushMatrix(); // glPushMatrix、glPopMatrix操作事实上就相当于栈里的入栈和出栈

    Sophus::Matrix4f m = Twc.matrix().template cast<float>();
    glMultMatrixf((GLfloat *)m.data());

    if(color == nullptr) {
        // glColor3f(1.0f, 0.0f, 0.0f);
        glColor3f(1.0f, 0.0f, 1.0f);
    } else {
        glColor3f(color[0], color[1], color[2]);
    }
    glLineWidth(line_width);
    glBegin(GL_LINES); // beigin
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 -cy) / fy, sz);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 -cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 -cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 -cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 -cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();  // end;
    glPopMatrix();
}

// --------------------------------------------------------------------------------------------------------------
void Viewer::DrawFrameResult(FrameResult::Ptr frame_result, const float *color) {
    SE3 Tcr = frame_result->RelativePose();
    SE3 Trw = frame_result->reference_keyframe_ptr_.lock()->Pose();
    SE3 Twc = (Tcr * Trw).inverse();

    // const float sz = 1.0f;
    const float sz = Viewer_sz;
    // const float sz = 0.10f;
    // const int line_width = 2;
    const float line_width = 2.0f;
    const float fx = 400.0f;
    const float fy = 400.0f;
    const float cx = 512.0f;
    const float cy = 384.0f;
    const float width = 1080.0f;
    const float height = 760.0f;

    glPushMatrix(); // glPushMatrix、glPopMatrix操作事实上就相当于栈里的入栈和出栈

    Sophus::Matrix4f m = Twc.matrix().template cast<float>();
    glMultMatrixf((GLfloat *)m.data());

    if(color == nullptr) {
        // glColor3f(1.0f, 0.0f, 0.0f);
        glColor3f(0.0f, 0.0f, 1.0f);
    } else {
        glColor3f(color[0], color[1], color[2]);
    }
    glLineWidth(line_width);
    glBegin(GL_LINES); // beigin
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 -cy) / fy, sz);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 -cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 -cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 -cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 -cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();  // end;
    glPopMatrix();
}

// --------------------------------------------------------------------------------------------------------------
void Viewer::DrawMapPointsAndFrameResult(const bool menuShowFrames, const bool menuShowPoints) {
    
    if(menuShowFrames) {
        for(auto &fr : frame_results_map_) {
            if(fr.second->ActiveKeyFlag()) {
                DrawFrameResult(fr.second, m_red_);
            } else if(fr.second->is_keyframe_) {
                DrawFrameResult(fr.second, m_green_);
            } else {
                DrawFrameResult(fr.second, m_blue_);
            }
        }

        // 20221028
        glColor3f(m_green_[0], m_green_[1], m_green_[2]);
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        for(unsigned long i=1; i < static_cast<unsigned long>(frame_results_map_.size()); i++) {
            Vec3 Oc = Vec3::Zero();
            auto cur_frame = frame_results_map_[i];
            SE3 Tcr = cur_frame->RelativePose();
            SE3 Trw = cur_frame->reference_keyframe_ptr_.lock()->Pose();
            SE3 Twc = (Tcr * Trw).inverse();
            Vec3 cur_frame_Ow = Twc * Oc;
            auto last_frame = frame_results_map_[i-1];
            Tcr = last_frame->RelativePose();
            Trw = last_frame->reference_keyframe_ptr_.lock()->Pose();
            Twc = (Tcr * Trw).inverse();
            Vec3 last_frame_Ow = Twc * Oc;
            glVertex3f(cur_frame_Ow[0], cur_frame_Ow[1], cur_frame_Ow[2]);
            glVertex3f(last_frame_Ow[0], last_frame_Ow[1], last_frame_Ow[2]);
        }
        glEnd();
    }

    if(menuShowPoints) {
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        for(auto &landmark : landmarks_map_) {
            auto pos = landmark.second->GetPos();
            if(landmark.second->active_observed_times_ == 0) {
                glColor3f(m_yellow_[0], m_yellow_[1], m_yellow_[2]);
            } else {
                glColor3f(m_red_[0], m_red_[1], m_red_[2]);
            }
            glVertex3d(pos[0], pos[1], pos[2]);
        }
        glEnd();
    }
}

// --------------------------------------------------------------------------------------------------------------
void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState &vis_camera) {
    SE3 Tcr = current_frame_->RelativePose();
    SE3 Trw = current_frame_->reference_keyframe_ptr_.lock()->Pose();
    SE3 Twc = (Tcr * Trw).inverse();

    pangolin::OpenGlMatrix m(Twc.matrix());
    vis_camera.Follow(m, true);
}

}  // namespace sg_slam

