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

#include "sensor_quality_assessor.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>

SensorQualityAssessor::SensorQualityAssessor()
    : lidar_density_threshold_(1000.0)
    , visual_feature_threshold_(200.0)
    , imu_freq_threshold_(200.0)
    , assessment_count_(0)
    , last_assessment_time_(0.0)
    , initialized_(false)
{
}

SensorQualityAssessor::~SensorQualityAssessor()
{
}

void SensorQualityAssessor::setAssessmentParameters(double lidar_density_threshold,
                                                   double visual_feature_threshold,
                                                   double imu_freq_threshold)
{
    lidar_density_threshold_ = lidar_density_threshold;
    visual_feature_threshold_ = visual_feature_threshold;
    imu_freq_threshold_ = imu_freq_threshold;
}

void SensorQualityAssessor::reset()
{
    while (!lidar_history_.empty()) lidar_history_.pop();
    while (!visual_history_.empty()) visual_history_.pop();
    while (!imu_history_.empty()) imu_history_.pop();
    
    assessment_count_ = 0;
    last_assessment_time_ = 0.0;
    initialized_ = false;
    prev_image_.release();
    prev_features_.clear();
}

// =============================================================================
// LiDAR质量评估实现
// =============================================================================

LidarQualityMetrics SensorQualityAssessor::assessLidarQuality(
    const PointCloudXYZI::Ptr& cloud,
    const std::vector<PointToPlane>& matches)
{
    LidarQualityMetrics metrics;
    
    if (!cloud || cloud->empty()) {
        return metrics; // 返回默认的低质量指标
    }
    
    // 1. 评估点云密度
    metrics.point_density_score = evaluatePointDensity(cloud);
    metrics.total_points = cloud->size();
    
    // 2. 评估几何一致性
    metrics.geometric_consistency_score = evaluateGeometricConsistency(matches);
    if (!matches.empty()) {
        double total_residual = 0.0;
        for (const auto& match : matches) {
            total_residual += std::abs(match.dis_to_plane_);
        }
        metrics.avg_residual = total_residual / matches.size();
    }
    
    // 3. 评估噪声水平
    metrics.noise_level_score = evaluateNoiseLevel(cloud);
    
    // 4. 评估特征丰富度
    metrics.feature_richness_score = evaluateFeatureRichness(cloud);
    
    // 5. 计算综合评分
    metrics.overall_score = weights_.lidar_density_weight * metrics.point_density_score +
                           weights_.lidar_consistency_weight * metrics.geometric_consistency_score +
                           weights_.lidar_noise_weight * metrics.noise_level_score +
                           weights_.lidar_richness_weight * metrics.feature_richness_score;
    
    // 确保评分在合理范围内
    metrics.overall_score = std::max(0.0, std::min(1.0, metrics.overall_score));
    
    // 更新历史记录
    lidar_history_.push(metrics);
    if (lidar_history_.size() > HISTORY_SIZE) {
        lidar_history_.pop();
    }
    
    return metrics;
}

double SensorQualityAssessor::evaluatePointDensity(const PointCloudXYZI::Ptr& cloud)
{
    if (!cloud || cloud->empty()) return 0.0;
    
    // 计算点云密度：点数/体积
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    double min_z = std::numeric_limits<double>::max();
    double max_z = std::numeric_limits<double>::lowest();
    
    for (const auto& point : *cloud) {
        min_x = std::min(min_x, static_cast<double>(point.x));
        max_x = std::max(max_x, static_cast<double>(point.x));
        min_y = std::min(min_y, static_cast<double>(point.y));
        max_y = std::max(max_y, static_cast<double>(point.y));
        min_z = std::min(min_z, static_cast<double>(point.z));
        max_z = std::max(max_z, static_cast<double>(point.z));
    }
    
    double volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z);
    if (volume < 1e-6) return 0.5; // 避免除零
    
    double density = cloud->size() / volume;
    
    // 使用sigmoid函数映射到[0,1]
    return sigmoid(density / lidar_density_threshold_);
}

double SensorQualityAssessor::evaluateGeometricConsistency(const std::vector<PointToPlane>& matches)
{
    if (matches.empty()) return 0.0;
    
    // 计算平面拟合残差的统计特性
    std::vector<double> residuals;
    residuals.reserve(matches.size());
    
    for (const auto& match : matches) {
        residuals.push_back(std::abs(match.dis_to_plane_));
    }
    
    // 计算残差的均值和标准差
    double mean_residual = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
    
    double variance = 0.0;
    for (double residual : residuals) {
        variance += (residual - mean_residual) * (residual - mean_residual);
    }
    variance /= residuals.size();
    double std_residual = std::sqrt(variance);
    
    // 基于残差大小评估一致性：残差越小，一致性越好
    double residual_score = 1.0 / (1.0 + mean_residual * 10.0); // 残差以米为单位
    
    // 基于残差分布评估一致性：标准差越小，一致性越好
    double consistency_score = 1.0 / (1.0 + std_residual * 20.0);
    
    return 0.6 * residual_score + 0.4 * consistency_score;
}

double SensorQualityAssessor::evaluateNoiseLevel(const PointCloudXYZI::Ptr& cloud)
{
    if (!cloud || cloud->size() < 10) return 0.5;
    
    // 计算相邻点的距离分布来评估噪声
    std::vector<double> neighbor_distances;
    neighbor_distances.reserve(cloud->size());
    
    for (size_t i = 0; i < cloud->size() - 1; ++i) {
        const auto& p1 = (*cloud)[i];
        const auto& p2 = (*cloud)[i + 1];
        
        double dist = std::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                               (p1.y - p2.y) * (p1.y - p2.y) +
                               (p1.z - p2.z) * (p1.z - p2.z));
        if (dist > 0.001 && dist < 5.0) { // 过滤明显异常的距离
            neighbor_distances.push_back(dist);
        }
    }
    
    if (neighbor_distances.empty()) return 0.5;
    
    // 计算距离分布的变异系数（标准差/均值）
    double mean_dist = std::accumulate(neighbor_distances.begin(), neighbor_distances.end(), 0.0) / neighbor_distances.size();
    
    double variance = 0.0;
    for (double dist : neighbor_distances) {
        variance += (dist - mean_dist) * (dist - mean_dist);
    }
    variance /= neighbor_distances.size();
    double std_dist = std::sqrt(variance);
    
    double coefficient_of_variation = (mean_dist > 0) ? (std_dist / mean_dist) : 1.0;
    
    // 变异系数越小，噪声越低，质量越高
    return 1.0 / (1.0 + coefficient_of_variation * 5.0);
}

double SensorQualityAssessor::evaluateFeatureRichness(const PointCloudXYZI::Ptr& cloud)
{
    if (!cloud || cloud->size() < 100) return 0.0;
    
    // 简化的特征丰富度评估：基于强度分布和空间分布
    std::vector<double> intensities;
    std::vector<double> ranges;
    
    intensities.reserve(cloud->size());
    ranges.reserve(cloud->size());
    
    for (const auto& point : *cloud) {
        intensities.push_back(point.intensity);
        double range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        ranges.push_back(range);
    }
    
    // 计算强度分布的方差（方差越大，特征越丰富）
    double mean_intensity = std::accumulate(intensities.begin(), intensities.end(), 0.0) / intensities.size();
    double intensity_variance = 0.0;
    for (double intensity : intensities) {
        intensity_variance += (intensity - mean_intensity) * (intensity - mean_intensity);
    }
    intensity_variance /= intensities.size();
    
    // 计算距离分布的方差
    double mean_range = std::accumulate(ranges.begin(), ranges.end(), 0.0) / ranges.size();
    double range_variance = 0.0;
    for (double range : ranges) {
        range_variance += (range - mean_range) * (range - mean_range);
    }
    range_variance /= ranges.size();
    
    // 归一化特征丰富度评分
    double intensity_score = sigmoid(intensity_variance / 1000.0); // 根据实际强度范围调整
    double range_score = sigmoid(range_variance / 100.0);         // 根据实际距离范围调整
    
    return 0.7 * intensity_score + 0.3 * range_score;
}

// =============================================================================
// 视觉质量评估实现
// =============================================================================

VisualQualityMetrics SensorQualityAssessor::assessVisualQuality(
    const cv::Mat& img,
    int feature_count,
    int match_count,
    const std::vector<cv::Point2f>& last_features)
{
    VisualQualityMetrics metrics;
    
    if (img.empty()) {
        return metrics; // 返回默认的低质量指标
    }
    
    // 1. 评估特征数量
    metrics.feature_count_score = evaluateFeatureCount(feature_count);
    metrics.feature_count = feature_count;
    
    // 2. 评估特征分布
    metrics.feature_distribution_score = evaluateFeatureDistribution(img, feature_count);
    
    // 3. 评估图像对比度
    metrics.contrast_score = evaluateImageContrast(img);
    
    // 4. 评估运动模糊
    metrics.motion_blur_score = evaluateMotionBlur(img);
    
    // 5. 评估跟踪稳定性
    metrics.tracking_stability_score = evaluateTrackingStability(match_count, feature_count);
    metrics.match_ratio = (feature_count > 0) ? static_cast<double>(match_count) / feature_count : 0.0;
    
    // 6. 评估光照条件
    metrics.illumination_score = evaluateIllumination(img);
    
    // 7. 计算综合评分
    metrics.overall_score = weights_.visual_count_weight * metrics.feature_count_score +
                           weights_.visual_distribution_weight * metrics.feature_distribution_score +
                           weights_.visual_contrast_weight * metrics.contrast_score +
                           weights_.visual_blur_weight * metrics.motion_blur_score +
                           weights_.visual_tracking_weight * metrics.tracking_stability_score +
                           weights_.visual_illumination_weight * metrics.illumination_score;
    
    // 确保评分在合理范围内
    metrics.overall_score = std::max(0.0, std::min(1.0, metrics.overall_score));
    
    // 更新历史记录
    visual_history_.push(metrics);
    if (visual_history_.size() > HISTORY_SIZE) {
        visual_history_.pop();
    }
    
    // 保存当前图像用于下一次评估
    prev_image_ = img.clone();
    
    return metrics;
}

double SensorQualityAssessor::evaluateFeatureCount(int feature_count)
{
    // 使用sigmoid函数将特征数量映射到[0,1]
    return sigmoid(static_cast<double>(feature_count) / visual_feature_threshold_);
}

double SensorQualityAssessor::evaluateFeatureDistribution(const cv::Mat& img, int feature_count)
{
    if (feature_count == 0 || img.empty()) return 0.0;
    
    // 简化评估：假设特征在图像中均匀分布是最好的
    // 实际应用中可以获取特征位置进行更精确的分布评估
    
    // 将图像分成网格，检查每个网格的特征分布
    int grid_rows = 4;
    int grid_cols = 4;
    int total_grids = grid_rows * grid_cols;
    
    // 估算每个网格应该有的特征数量
    double expected_features_per_grid = static_cast<double>(feature_count) / total_grids;
    
    // 如果特征数量很少，降低分布要求
    if (expected_features_per_grid < 1.0) {
        return sigmoid(feature_count / 50.0); // 特征太少时主要考虑数量
    }
    
    // 假设特征分布相对均匀，给出中等评分
    // 实际实现中需要获取特征坐标进行精确计算
    return 0.7; // 中等分布质量
}

double SensorQualityAssessor::evaluateImageContrast(const cv::Mat& img)
{
    if (img.empty()) return 0.0;
    
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    // 计算图像的标准差作为对比度指标
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    
    double contrast = stddev[0];
    
    // 将对比度映射到[0,1]，标准差越大对比度越好
    return sigmoid(contrast / 50.0); // 50是经验阈值
}

double SensorQualityAssessor::evaluateMotionBlur(const cv::Mat& img)
{
    if (img.empty()) return 0.5;
    
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    // 使用Laplacian算子检测图像清晰度
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    double sharpness = stddev[0] * stddev[0]; // 方差越大越清晰
    
    // 将清晰度映射到[0,1]
    return sigmoid(sharpness / 1000.0); // 1000是经验阈值
}

double SensorQualityAssessor::evaluateTrackingStability(int match_count, int total_count)
{
    if (total_count == 0) return 0.0;
    
    double match_ratio = static_cast<double>(match_count) / total_count;
    
    // 匹配率越高，跟踪越稳定
    return sigmoid(match_ratio * 5.0 - 2.5); // 调整sigmoid的中心点
}

double SensorQualityAssessor::evaluateIllumination(const cv::Mat& img)
{
    if (img.empty()) return 0.5;
    
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    // 计算图像亮度分布
    cv::Scalar mean_brightness = cv::mean(gray);
    double brightness = mean_brightness[0];
    
    // 理想亮度范围是[80, 180]（8位图像）
    double ideal_min = 80.0;
    double ideal_max = 180.0;
    double ideal_center = (ideal_min + ideal_max) / 2.0;
    
    // 计算与理想亮度的偏差
    double brightness_deviation = std::abs(brightness - ideal_center) / (ideal_max - ideal_min);
    
    // 偏差越小，光照质量越好
    return 1.0 - std::min(1.0, brightness_deviation);
}

// =============================================================================
// IMU质量评估实现
// =============================================================================

ImuQualityMetrics SensorQualityAssessor::assessImuQuality(
    const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr>& imu_buf)
{
    ImuQualityMetrics metrics;
    
    if (imu_buf.size() < 2) {
        return metrics; // 返回默认质量指标
    }
    
    // 1. 评估加速度计一致性
    metrics.acceleration_consistency_score = evaluateAccelerationConsistency(imu_buf);
    
    // 2. 评估陀螺仪一致性
    metrics.gyro_consistency_score = evaluateGyroConsistency(imu_buf);
    
    // 3. 评估偏置稳定性
    metrics.bias_stability_score = evaluateBiasStability(imu_buf);
    
    // 4. 评估频率稳定性
    metrics.frequency_score = evaluateFrequencyStability(imu_buf);
    
    // 5. 计算综合评分
    metrics.overall_score = weights_.imu_acc_weight * metrics.acceleration_consistency_score +
                           weights_.imu_gyro_weight * metrics.gyro_consistency_score +
                           weights_.imu_bias_weight * metrics.bias_stability_score +
                           weights_.imu_freq_weight * metrics.frequency_score;
    
    // 确保评分在合理范围内
    metrics.overall_score = std::max(0.0, std::min(1.0, metrics.overall_score));
    
    // 更新历史记录
    imu_history_.push(metrics);
    if (imu_history_.size() > HISTORY_SIZE) {
        imu_history_.pop();
    }
    
    return metrics;
}

double SensorQualityAssessor::evaluateAccelerationConsistency(
    const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr>& imu_buf)
{
    if (imu_buf.size() < 10) return 0.8; // 数据不足时给默认值
    
    std::vector<double> acc_magnitudes;
    acc_magnitudes.reserve(imu_buf.size());
    
    for (const auto& imu_msg : imu_buf) {
        double acc_x = imu_msg->linear_acceleration.x;
        double acc_y = imu_msg->linear_acceleration.y;
        double acc_z = imu_msg->linear_acceleration.z;
        
        double magnitude = std::sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z);
        acc_magnitudes.push_back(magnitude);
    }
    
    // 计算加速度幅值的变异系数
    double mean_acc = std::accumulate(acc_magnitudes.begin(), acc_magnitudes.end(), 0.0) / acc_magnitudes.size();
    
    double variance = 0.0;
    for (double acc : acc_magnitudes) {
        variance += (acc - mean_acc) * (acc - mean_acc);
    }
    variance /= acc_magnitudes.size();
    double std_acc = std::sqrt(variance);
    
    double coefficient_of_variation = (mean_acc > 0) ? (std_acc / mean_acc) : 1.0;
    
    // 变异系数越小，一致性越好
    return 1.0 / (1.0 + coefficient_of_variation * 10.0);
}

double SensorQualityAssessor::evaluateGyroConsistency(
    const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr>& imu_buf)
{
    if (imu_buf.size() < 10) return 0.8; // 数据不足时给默认值
    
    std::vector<double> gyro_magnitudes;
    gyro_magnitudes.reserve(imu_buf.size());
    
    for (const auto& imu_msg : imu_buf) {
        double gyro_x = imu_msg->angular_velocity.x;
        double gyro_y = imu_msg->angular_velocity.y;
        double gyro_z = imu_msg->angular_velocity.z;
        
        double magnitude = std::sqrt(gyro_x * gyro_x + gyro_y * gyro_y + gyro_z * gyro_z);
        gyro_magnitudes.push_back(magnitude);
    }
    
    // 计算角速度的方差
    double mean_gyro = std::accumulate(gyro_magnitudes.begin(), gyro_magnitudes.end(), 0.0) / gyro_magnitudes.size();
    
    double variance = 0.0;
    for (double gyro : gyro_magnitudes) {
        variance += (gyro - mean_gyro) * (gyro - mean_gyro);
    }
    variance /= gyro_magnitudes.size();
    
    // 方差越小，一致性越好
    return 1.0 / (1.0 + variance * 100.0);
}

double SensorQualityAssessor::evaluateBiasStability(
    const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr>& imu_buf)
{
    if (imu_buf.size() < 20) return 0.8; // 数据不足时给默认值
    
    // 简化的偏置稳定性评估：检查长期趋势
    double first_half_acc_x = 0.0, first_half_acc_y = 0.0, first_half_acc_z = 0.0;
    double second_half_acc_x = 0.0, second_half_acc_y = 0.0, second_half_acc_z = 0.0;
    double first_half_gyro_x = 0.0, first_half_gyro_y = 0.0, first_half_gyro_z = 0.0;
    double second_half_gyro_x = 0.0, second_half_gyro_y = 0.0, second_half_gyro_z = 0.0;
    
    size_t half_size = imu_buf.size() / 2;
    
    // 计算前半部分平均值
    for (size_t i = 0; i < half_size; ++i) {
        const auto& imu_msg = imu_buf[i];
        first_half_acc_x += imu_msg->linear_acceleration.x;
        first_half_acc_y += imu_msg->linear_acceleration.y;
        first_half_acc_z += imu_msg->linear_acceleration.z;
        first_half_gyro_x += imu_msg->angular_velocity.x;
        first_half_gyro_y += imu_msg->angular_velocity.y;
        first_half_gyro_z += imu_msg->angular_velocity.z;
    }
    first_half_acc_x /= half_size;
    first_half_acc_y /= half_size;
    first_half_acc_z /= half_size;
    first_half_gyro_x /= half_size;
    first_half_gyro_y /= half_size;
    first_half_gyro_z /= half_size;
    
    // 计算后半部分平均值
    for (size_t i = half_size; i < imu_buf.size(); ++i) {
        const auto& imu_msg = imu_buf[i];
        second_half_acc_x += imu_msg->linear_acceleration.x;
        second_half_acc_y += imu_msg->linear_acceleration.y;
        second_half_acc_z += imu_msg->linear_acceleration.z;
        second_half_gyro_x += imu_msg->angular_velocity.x;
        second_half_gyro_y += imu_msg->angular_velocity.y;
        second_half_gyro_z += imu_msg->angular_velocity.z;
    }
    size_t second_half_size = imu_buf.size() - half_size;
    second_half_acc_x /= second_half_size;
    second_half_acc_y /= second_half_size;
    second_half_acc_z /= second_half_size;
    second_half_gyro_x /= second_half_size;
    second_half_gyro_y /= second_half_size;
    second_half_gyro_z /= second_half_size;
    
    // 计算前后半部分的差异
    double acc_drift = std::sqrt((first_half_acc_x - second_half_acc_x) * (first_half_acc_x - second_half_acc_x) +
                                (first_half_acc_y - second_half_acc_y) * (first_half_acc_y - second_half_acc_y) +
                                (first_half_acc_z - second_half_acc_z) * (first_half_acc_z - second_half_acc_z));
    
    double gyro_drift = std::sqrt((first_half_gyro_x - second_half_gyro_x) * (first_half_gyro_x - second_half_gyro_x) +
                                 (first_half_gyro_y - second_half_gyro_y) * (first_half_gyro_y - second_half_gyro_y) +
                                 (first_half_gyro_z - second_half_gyro_z) * (first_half_gyro_z - second_half_gyro_z));
    
    double total_drift = acc_drift + gyro_drift * 10.0; // 陀螺仪漂移权重更高
    
    // 漂移越小，稳定性越好
    return 1.0 / (1.0 + total_drift * 1000.0);
}

double SensorQualityAssessor::evaluateFrequencyStability(
    const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr>& imu_buf)
{
    if (imu_buf.size() < 10) return 0.8; // 数据不足时给默认值
    
    // 计算时间间隔
    std::vector<double> time_intervals;
    time_intervals.reserve(imu_buf.size() - 1);
    
    for (size_t i = 1; i < imu_buf.size(); ++i) {
        double t1 = imu_buf[i-1]->header.stamp.sec + imu_buf[i-1]->header.stamp.nanosec * 1e-9;
        double t2 = imu_buf[i]->header.stamp.sec + imu_buf[i]->header.stamp.nanosec * 1e-9;
        
        double interval = t2 - t1;
        if (interval > 0.001 && interval < 0.1) { // 过滤异常间隔
            time_intervals.push_back(interval);
        }
    }
    
    if (time_intervals.empty()) return 0.5;
    
    // 计算时间间隔的变异系数
    double mean_interval = std::accumulate(time_intervals.begin(), time_intervals.end(), 0.0) / time_intervals.size();
    
    double variance = 0.0;
    for (double interval : time_intervals) {
        variance += (interval - mean_interval) * (interval - mean_interval);
    }
    variance /= time_intervals.size();
    double std_interval = std::sqrt(variance);
    
    double coefficient_of_variation = (mean_interval > 0) ? (std_interval / mean_interval) : 1.0;
    
    // 变异系数越小，频率越稳定
    return 1.0 / (1.0 + coefficient_of_variation * 100.0);
}

// =============================================================================
// 工具函数实现
// =============================================================================

double SensorQualityAssessor::sigmoid(double x, double alpha, double beta)
{
    return 1.0 / (1.0 + std::exp(-alpha * (x - beta)));
}

double SensorQualityAssessor::normalizeScore(double raw_value, double min_val, double max_val, bool invert)
{
    if (max_val <= min_val) return 0.5;
    
    double normalized = (raw_value - min_val) / (max_val - min_val);
    normalized = std::max(0.0, std::min(1.0, normalized));
    
    return invert ? (1.0 - normalized) : normalized;
}
