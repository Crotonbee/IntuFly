# 运行说明

本项目建议在ubuntu22.04、Gazebo Sim 8.9.0的harmonic版本以及ros2 humble版本下运行，另外请先下载好PX4源码。  
在ubuntu22.04中使用v1.14+版本PX4，配置PX4自动下载的gzaebo无法通过桥接包与ROS2通信，需要手动桥接。

## 手动桥接

1. 设置环境变量

```bash
export GZ_VERSION=harmonic
