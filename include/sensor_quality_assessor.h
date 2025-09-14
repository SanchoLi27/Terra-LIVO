/*
 * TERRA-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry
 * Copyright (C) 2025 SanchoLi27 <hdalhd1104@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef SENSOR_QUALITY_ASSESSOR_H
#define SENSOR_QUALITY_ASSESSOR_H

#include "common_lib.h"
#include "voxel_map.h"
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <deque>
#include <vector>
#include <queue>

/**
 * @brief LiDAR质量评估指标结构体
 */
struct LidarQualityMetrics
{
    double point_density_score;         // 点云密度评分 [0,1]
    double geometric_consistency_score;  // 几何一致性评分 [0,1]
    double noise_level_score;           // 噪声水平评分 [0,1]
    double feature_richness_score;      // 特征丰富度评分 [0,1]
    double overall_score;               // 综合评分 [0,1]
    
    // 原始数据用于调试
    int total_points;
    double avg_residual;
    double noise_variance;
    int plane_count;
    int edge_count;
    
    LidarQualityMetrics() : point_density_score(0.5), geometric_consistency_score(0.5),
                           noise_level_score(0.5), feature_richness_score(0.5),
                           overall_score(0.5), total_points(0), avg_residual(0.0),
                           noise_variance(0.0), plane_count(0), edge_count(0) {}
};

/**
 * @brief 视觉质量评估指标结构体
 */
struct VisualQualityMetrics
{
    double feature_count_score;         // 特征数量评分 [0,1]
    double feature_distribution_score;  // 特征分布评分 [0,1]
    double contrast_score;              // 图像对比度评分 [0,1]
    double motion_blur_score;           // 运动模糊评分 [0,1]
    double tracking_stability_score;    // 跟踪稳定性评分 [0,1]
    double illumination_score;          // 光照条件评分 [0,1]
    double overall_score;               // 综合评分 [0,1]
    
    // 原始数据用于调试
    int feature_count;
    double distribution_variance;
    double contrast_value;
    double blur_metric;
    double match_ratio;
    double brightness_variance;
    
    VisualQualityMetrics() : feature_count_score(0.5), feature_distribution_score(0.5),
                            contrast_score(0.5), motion_blur_score(0.5),
                            tracking_stability_score(0.5), illumination_score(0.5),
                            overall_score(0.5), feature_count(0), distribution_variance(0.0),
                            contrast_value(0.0), blur_metric(0.0), match_ratio(0.0),
                            brightness_variance(0.0) {}
};

/**
 * @brief IMU质量评估指标结构体
 */
struct ImuQualityMetrics
{
    double acceleration_consistency_score; // 加速度一致性评分 [0,1]
    double gyro_consistency_score;        // 陀螺仪一致性评分 [0,1]
    double bias_stability_score;          // 偏置稳定性评分 [0,1]
    double frequency_score;               // 频率稳定性评分 [0,1]
    double overall_score;                 // 综合评分 [0,1]
    
    // 原始数据用于调试
    double acc_variance;
    double gyro_variance;
    double bias_drift;
    double freq_deviation;
    
    ImuQualityMetrics() : acceleration_consistency_score(0.8), gyro_consistency_score(0.8),
                         bias_stability_score(0.8), frequency_score(0.8), overall_score(0.8),
                         acc_variance(0.0), gyro_variance(0.0), bias_drift(0.0),
                         freq_deviation(0.0) {}
};

/**
 * @brief 传感器质量评估器
 * 负责实时评估LiDAR、视觉、IMU传感器的数据质量
 */
class SensorQualityAssessor
{
public:
    SensorQualityAssessor();
    ~SensorQualityAssessor();
    
    /**
     * @brief 评估LiDAR数据质量
     * @param cloud 输入点云
     * @param matches 点到平面匹配结果
     * @return LiDAR质量指标
     */
    LidarQualityMetrics assessLidarQuality(const PointCloudXYZI::Ptr& cloud,
                                          const std::vector<PointToPlane>& matches);
    
    /**
     * @brief 评估视觉数据质量
     * @param img 输入图像
     * @param feature_count 当前帧特征数量
     * @param match_count 成功匹配的特征数量
     * @param last_features 上一帧特征位置（用于运动估计）
     * @return 视觉质量指标
     */
    VisualQualityMetrics assessVisualQuality(const cv::Mat& img,
                                            int feature_count,
                                            int match_count,
                                            const std::vector<cv::Point2f>& last_features = {});
    
    /**
     * @brief 评估IMU数据质量
     * @param imu_buf IMU数据缓冲区
     * @return IMU质量指标
     */
    ImuQualityMetrics assessImuQuality(const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr>& imu_buf);
    
    /**
     * @brief 重置评估器状态
     */
    void reset();
    
    /**
     * @brief 设置评估参数
     */
    void setAssessmentParameters(double lidar_density_threshold = 1000.0,
                               double visual_feature_threshold = 200.0,
                               double imu_freq_threshold = 200.0);

private:
    // LiDAR质量评估私有方法
    double evaluatePointDensity(const PointCloudXYZI::Ptr& cloud);
    double evaluateGeometricConsistency(const std::vector<PointToPlane>& matches);
    double evaluateNoiseLevel(const PointCloudXYZI::Ptr& cloud);
    double evaluateFeatureRichness(const PointCloudXYZI::Ptr& cloud);
    
    // 视觉质量评估私有方法
    double evaluateFeatureCount(int feature_count);
    double evaluateFeatureDistribution(const cv::Mat& img, int feature_count);
    double evaluateImageContrast(const cv::Mat& img);
    double evaluateMotionBlur(const cv::Mat& img);
    double evaluateTrackingStability(int match_count, int total_count);
    double evaluateIllumination(const cv::Mat& img);
    
    // IMU质量评估私有方法
    double evaluateAccelerationConsistency(const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr>& imu_buf);
    double evaluateGyroConsistency(const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr>& imu_buf);
    double evaluateBiasStability(const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr>& imu_buf);
    double evaluateFrequencyStability(const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr>& imu_buf);
    
    // 工具函数
    double sigmoid(double x, double alpha = 2.0, double beta = 0.5);
    double normalizeScore(double raw_value, double min_val, double max_val, bool invert = false);
    
private:
    // 参数配置
    double lidar_density_threshold_;
    double visual_feature_threshold_;
    double imu_freq_threshold_;
    
    // 历史数据（用于趋势分析）
    std::queue<LidarQualityMetrics> lidar_history_;
    std::queue<VisualQualityMetrics> visual_history_;
    std::queue<ImuQualityMetrics> imu_history_;
    
    static const int HISTORY_SIZE = 10;
    
    // 统计数据
    int assessment_count_;
    double last_assessment_time_;
    
    // 内部状态
    bool initialized_;
    cv::Mat prev_image_;
    std::vector<cv::Point2f> prev_features_;
    
    // 权重参数（用于综合评分计算）
    struct WeightParams {
        // LiDAR权重
        double lidar_density_weight = 0.3;
        double lidar_consistency_weight = 0.3;
        double lidar_noise_weight = 0.2;
        double lidar_richness_weight = 0.2;
        
        // 视觉权重
        double visual_count_weight = 0.25;
        double visual_distribution_weight = 0.2;
        double visual_contrast_weight = 0.2;
        double visual_blur_weight = 0.15;
        double visual_tracking_weight = 0.15;
        double visual_illumination_weight = 0.05;
        
        // IMU权重
        double imu_acc_weight = 0.3;
        double imu_gyro_weight = 0.3;
        double imu_bias_weight = 0.2;
        double imu_freq_weight = 0.2;
    } weights_;
};

#endif // SENSOR_QUALITY_ASSESSOR_H
