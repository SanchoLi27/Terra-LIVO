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

#ifndef ADAPTIVE_WEIGHT_CALCULATOR_H
#define ADAPTIVE_WEIGHT_CALCULATOR_H

#include "sensor_quality_assessor.h"
#include "common_lib.h"
#include <queue>
#include <vector>
#include <Eigen/Dense>

/**
 * @brief 传感器融合权重结构体
 */
struct SensorFusionWeights
{
    double lidar_weight;     // LiDAR权重 [0,1]
    double visual_weight;    // 视觉权重 [0,1]
    double imu_weight;       // IMU权重 [0,1]
    double total_confidence; // 总体置信度 [0,1]
    
    // 权重变化率（用于稳定性分析）
    double lidar_change_rate;
    double visual_change_rate;
    double imu_change_rate;
    
    // 时间戳
    double timestamp;
    
    SensorFusionWeights() : lidar_weight(0.4), visual_weight(0.4), imu_weight(0.2),
                           total_confidence(0.5), lidar_change_rate(0.0),
                           visual_change_rate(0.0), imu_change_rate(0.0),
                           timestamp(0.0) {}
    
    SensorFusionWeights(double l_w, double v_w, double i_w, double conf = 0.5) 
        : lidar_weight(l_w), visual_weight(v_w), imu_weight(i_w),
          total_confidence(conf), lidar_change_rate(0.0), visual_change_rate(0.0),
          imu_change_rate(0.0), timestamp(0.0) {}
    
    // 权重归一化
    void normalize() {
        double sum = lidar_weight + visual_weight + imu_weight;
        if (sum > 1e-6) {
            lidar_weight /= sum;
            visual_weight /= sum;
            imu_weight /= sum;
        } else {
            lidar_weight = visual_weight = imu_weight = 1.0 / 3.0;
        }
    }
    
    // 权重有效性检查
    bool isValid() const {
        return (lidar_weight >= 0.0 && lidar_weight <= 1.0 &&
                visual_weight >= 0.0 && visual_weight <= 1.0 &&
                imu_weight >= 0.0 && imu_weight <= 1.0 &&
                std::abs(lidar_weight + visual_weight + imu_weight - 1.0) < 1e-3);
    }
};

/**
 * @brief 自适应权重计算器
 * 基于传感器质量评估结果，计算最优的传感器融合权重
 */
class AdaptiveWeightCalculator
{
public:
    /**
     * @brief 权重计算策略枚举
     */
    enum WeightStrategy {
        QUALITY_BASED,          // 基于质量的权重分配
        CONFIDENCE_BASED,       // 基于置信度的权重分配
        HYBRID,                 // 混合策略
        CONSERVATIVE            // 保守策略（变化较小）
    };
    
    /**
     * @brief 权重计算参数结构体
     */
    struct CalculationParams {
        // 基础参数
        double quality_sensitivity = 2.0;      // 质量敏感度
        double min_weight_threshold = 0.05;    // 最小权重阈值
        double max_weight_threshold = 0.85;    // 最大权重阈值
        double smoothing_factor = 0.8;         // 平滑因子
        
        // IMU基础权重（IMU作为基础约束）
        double imu_base_weight = 0.15;
        double imu_max_weight = 0.7;
        
        // 权重变化限制
        double max_weight_change_rate = 0.1;   // 单次最大权重变化
        double stability_threshold = 0.02;     // 稳定性阈值
        
        // 置信度参数
        double confidence_alpha = 1.5;         // 置信度映射参数
        double confidence_beta = 0.3;          // 置信度最小值
        
        // 异常检测参数
        double outlier_detection_threshold = 2.0; // 异常值检测阈值（标准差倍数）
        bool enable_outlier_rejection = true;     // 是否启用异常值拒绝
        
        CalculationParams() = default;
    };

public:
    AdaptiveWeightCalculator();
    explicit AdaptiveWeightCalculator(const CalculationParams& params);
    ~AdaptiveWeightCalculator();
    
    /**
     * @brief 计算自适应融合权重
     * @param lidar_quality LiDAR质量指标
     * @param visual_quality 视觉质量指标
     * @param imu_quality IMU质量指标
     * @param strategy 权重计算策略
     * @return 融合权重结构体
     */
    SensorFusionWeights calculateAdaptiveWeights(
        const LidarQualityMetrics& lidar_quality,
        const VisualQualityMetrics& visual_quality,
        const ImuQualityMetrics& imu_quality,
        WeightStrategy strategy = HYBRID
    );
    
    /**
     * @brief 设置计算参数
     * @param params 新的计算参数
     */
    void setCalculationParams(const CalculationParams& params);
    
    /**
     * @brief 获取当前计算参数
     * @return 当前计算参数
     */
    const CalculationParams& getCalculationParams() const { return params_; }
    
    /**
     * @brief 重置计算器状态
     */
    void reset();
    
    /**
     * @brief 获取权重历史
     * @return 权重历史队列的拷贝
     */
    std::vector<SensorFusionWeights> getWeightHistory() const;
    
    /**
     * @brief 获取权重稳定性指标
     * @return 权重稳定性评分 [0,1]
     */
    double getWeightStability() const;
    
    /**
     * @brief 预测下一时刻的权重
     * @param prediction_horizon 预测时间范围
     * @return 预测的权重
     */
    SensorFusionWeights predictNextWeights(double prediction_horizon = 0.1) const;
    
    /**
     * @brief 设置权重变化回调函数
     * @param callback 回调函数
     */
    void setWeightChangeCallback(std::function<void(const SensorFusionWeights&)> callback);

private:
    // 核心权重计算方法
    SensorFusionWeights calculateQualityBasedWeights(
        const LidarQualityMetrics& lidar_quality,
        const VisualQualityMetrics& visual_quality,
        const ImuQualityMetrics& imu_quality
    );
    
    SensorFusionWeights calculateConfidenceBasedWeights(
        const LidarQualityMetrics& lidar_quality,
        const VisualQualityMetrics& visual_quality,
        const ImuQualityMetrics& imu_quality
    );
    
    SensorFusionWeights calculateHybridWeights(
        const LidarQualityMetrics& lidar_quality,
        const VisualQualityMetrics& visual_quality,
        const ImuQualityMetrics& imu_quality
    );
    
    SensorFusionWeights calculateConservativeWeights(
        const LidarQualityMetrics& lidar_quality,
        const VisualQualityMetrics& visual_quality,
        const ImuQualityMetrics& imu_quality
    );
    
    // 权重处理方法
    SensorFusionWeights applyConstraints(const SensorFusionWeights& weights) const;
    SensorFusionWeights applyTemporalSmoothing(const SensorFusionWeights& weights);
    SensorFusionWeights applyStabilityControl(const SensorFusionWeights& weights);
    SensorFusionWeights applyQualityBasedAdjustments(
        const SensorFusionWeights& weights,
        const LidarQualityMetrics& lidar_quality,
        const VisualQualityMetrics& visual_quality,
        const ImuQualityMetrics& imu_quality
    );
    
    // 工具方法
    double qualityToWeight(double quality_score, double sensitivity = 2.0);
    double calculateTotalConfidence(const LidarQualityMetrics& lidar_quality,
                                   const VisualQualityMetrics& visual_quality,
                                   const ImuQualityMetrics& imu_quality);
    bool isOutlier(const SensorFusionWeights& weights) const;
    void updateStatistics(const SensorFusionWeights& weights);
    
    // 数学工具
    double sigmoid(double x, double alpha = 2.0, double center = 0.5);
    double exponentialSmoothing(double current, double previous, double alpha);
    Eigen::Vector3d normalizeWeights(const Eigen::Vector3d& weights);

private:
    // 配置参数
    CalculationParams params_;
    
    // 权重历史（用于平滑和趋势分析）
    std::queue<SensorFusionWeights> weight_history_;
    static const int MAX_HISTORY_SIZE = 20;
    
    // 统计信息
    struct Statistics {
        double mean_lidar_weight = 0.4;
        double mean_visual_weight = 0.4;
        double mean_imu_weight = 0.2;
        double std_lidar_weight = 0.1;
        double std_visual_weight = 0.1;
        double std_imu_weight = 0.05;
        int sample_count = 0;
    } stats_;
    
    // 内部状态
    bool initialized_;
    SensorFusionWeights last_weights_;
    double last_calculation_time_;
    
    // 回调函数
    std::function<void(const SensorFusionWeights&)> weight_change_callback_;
    
    // 调试信息
    struct DebugInfo {
        bool enable_debug = false;
        std::string debug_output_path = "";
        std::vector<std::string> debug_log;
    } debug_info_;
    
public:
    // 调试和监控接口
    void enableDebugMode(bool enable, const std::string& output_path = "");
    void logDebugInfo(const std::string& info);
    std::vector<std::string> getDebugLog() const { return debug_info_.debug_log; }
    void clearDebugLog() { debug_info_.debug_log.clear(); }
};

#endif // ADAPTIVE_WEIGHT_CALCULATOR_H
