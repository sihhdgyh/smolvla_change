import gymnasium as gym
import cv2

# 创建环境（注意 render_mode）
env = gym.make("metaworld/basketball-v3", render_mode="rgb_array")
env.reset()
img = env.render() # 获取渲染帧

# 将 RGB 转为 BGR 保存查看
cv2.imwrite("check_render.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("检查 check_render.png，看物体是否出现，是否存在透视。")