//
//  Created by Lucifer on 2022/7/10.
//

#include "sg_slam/config.h"
#include <glog/logging.h>  // Google glog——一个基于程序级记录日志信息的c++库

namespace sg_slam {

std::shared_ptr<Config> Config::config_ = nullptr;

bool Config::SetParameterFile(const std::string &filename) {
    if(config_ == nullptr) {
        config_ = std::shared_ptr<Config>(new Config);
    }

    // 存储或读取数据的文件名（字符串），其扩展名(.xml 或 .yml/.yaml)决定文件格式。
    config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);

    if(config_->file_.isOpened() == false) {
        // FLAGS_log_dir = "./log";

        google::InitGoogleLogging("config");
        google::SetLogDestination(google::GLOG_ERROR, "./log/config/");
        // google::SetStderrLogging(google::GLOG_ERROR);
        LOG(ERROR) << "parameter file " << filename << " does not exist. ";
        google::ShutdownGoogleLogging();
        config_->file_.release();
        return false;
    }
    return true;
}

Config::~Config() {
    if(file_.isOpened()) {
        file_.release();
    }
}

}  // namespace sg_slam
