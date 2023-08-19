//
//  Created by Lucifer on 2022/7/10.
//

#ifndef _SG_SLAM_CONFIG_H_
#define _SG_SLAM_CONFIG_H_

#include <memory>  // 智能指针
#include <string>

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

namespace sg_slam {

/**
 * @brief 配置类，使用SetParameterFile确定配置文件, 然后用Get得到对应值
 * 单例模式: https://blog.csdn.net/m0_46655373/article/details/123932996
 */
class Config 
{
private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;

    Config() {}  // private constructor makes a singleton

public:
    ~Config();  // close the file when deconstructing

    // set a new config file
    static bool SetParameterFile(const std::string &filename);

    // access the parameter values
    // https://wenku.baidu.com/view/40a4e16bf4ec4afe04a1b0717fd5360cba1a8dd9.html
    // 模板定义最好放在头文件中.
    template <typename T>
    static T Get(const std::string &key) {
        return T(Config::config_->file_[key]);
    }

    static cv::FileNode GetCVmat(const std::string &key) {
        return Config::config_->file_[key];
    }

};  // class Config 

}  // namespace sg_slam

#endif  // _SG_SLAM_CONFIG_H_
