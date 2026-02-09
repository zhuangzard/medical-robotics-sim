# 医疗机器人仿真平台原型开发计划

**版本**: v1.0  
**计划周期**: 9 个月（2026年2月 - 2026年10月）  
**开发模式**: 敏捷迭代 + 风险驱动

---

## 📋 执行摘要

本计划将医疗机器人仿真平台开发分为**3个阶段 + 9个冲刺**：

**阶段1: 基础验证** (Month 1-3) - 证明技术可行性  
**阶段2: 核心集成** (Month 4-6) - 构建完整系统  
**阶段3: 临床优化** (Month 7-9) - 医疗场景定制

**关键里程碑**:
- ✅ Sprint 3: 软组织-刚体混合仿真 Demo
- ✅ Sprint 6: 多模态融合原型
- ✅ Sprint 9: 临床验证就绪版本

---

## 🎯 阶段1: 基础验证（Month 1-3）

### 目标
证明 Dynami-CAL + SOFA + MuJoCo 混合架构的技术可行性

### Sprint 1: 环境搭建与刚体基础（Week 1-4）

#### 任务清单
- [ ] **Task 1.1**: 开发环境配置
  ```bash
  # 系统依赖安装
  sudo apt install build-essential cmake git
  
  # 创建 Conda 环境
  conda create -n medsim python=3.10
  conda activate medsim
  
  # 安装核心库
  pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install torch-geometric pyg_lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
  pip install mujoco==3.1.0
  pip install open3d trimesh pyvista
  ```

- [ ] **Task 1.2**: MuJoCo 手术机器人模型
  ```xml
  <!-- surgical_robot.xml -->
  <mujoco model="surgical_robot">
    <option timestep="0.002" iterations="50"/>
    <worldbody>
      <body name="base">
        <geom type="cylinder" size="0.05 0.02" rgba="0.3 0.3 0.3 1"/>
        
        <!-- Shoulder joint -->
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" 
               range="-180 180" damping="0.5"/>
        <body name="upper_arm" pos="0 0 0.1">
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.3" 
                rgba="0.7 0.7 0.7 1"/>
          
          <!-- Elbow joint -->
          <joint name="elbow" type="hinge" axis="0 1 0" 
                 range="-120 120" damping="0.3"/>
          <body name="forearm" pos="0 0 0.3">
            <geom type="capsule" size="0.015" fromto="0 0 0 0 0 0.25" 
                  rgba="0.6 0.6 0.6 1"/>
            
            <!-- Wrist -->
            <joint name="wrist" type="hinge" axis="1 0 0" 
                   range="-90 90" damping="0.2"/>
            <body name="tool" pos="0 0 0.25">
              <!-- Grasper jaws -->
              <geom name="jaw_left" type="box" size="0.01 0.002 0.03" 
                    pos="0.005 0 0" rgba="0.8 0.2 0.2 1"/>
              <geom name="jaw_right" type="box" size="0.01 0.002 0.03" 
                    pos="-0.005 0 0" rgba="0.8 0.2 0.2 1"/>
              <joint name="gripper" type="slide" axis="1 0 0" 
                     range="0 0.01" damping="0.1"/>
            </body>
          </body>
        </body>
      </body>
    </worldbody>
    
    <actuator>
      <motor joint="shoulder_pan" ctrllimited="true" ctrlrange="-10 10"/>
      <motor joint="elbow" ctrllimited="true" ctrlrange="-5 5"/>
      <motor joint="wrist" ctrllimited="true" ctrlrange="-2 2"/>
      <motor joint="gripper" ctrllimited="true" ctrlrange="0 5"/>
    </actuator>
  </mujoco>
  ```

- [ ] **Task 1.3**: 基础仿真循环
  ```python
  import mujoco
  import numpy as np
  
  class SurgicalRobotSim:
      def __init__(self, model_path):
          self.model = mujoco.MjModel.from_xml_path(model_path)
          self.data = mujoco.MjData(self.model)
          
      def step(self, action):
          """
          action: [shoulder, elbow, wrist, gripper] joint torques
          """
          self.data.ctrl[:] = action
          mujoco.mj_step(self.model, self.data)
          
          # 返回状态
          state = {
              'qpos': self.data.qpos.copy(),
              'qvel': self.data.qvel.copy(),
              'tool_pos': self.data.body('tool').xpos.copy()
          }
          return state
      
      def reset(self):
          mujoco.mj_resetData(self.model, self.data)
  ```

- [ ] **Task 1.4**: 可视化与调试工具
  ```python
  from mujoco import viewer
  
  # 交互式查看器
  viewer.launch(model, data)
  ```

#### 验收标准
- ✅ 机器人模型在 MuJoCo 中流畅运动
- ✅ 关节控制响应正常
- ✅ 碰撞检测工作
- ✅ 运行速度 > 100 FPS

#### 风险与缓解
| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| MuJoCo 学习曲线陡峭 | 中 | 中 | 提前学习官方教程 |
| 模型不稳定 | 低 | 高 | 调整 damping/armature |

---

### Sprint 2: SOFA 软组织基础（Week 5-8）

#### 任务清单
- [ ] **Task 2.1**: SOFA 编译安装
  ```bash
  git clone https://github.com/sofa-framework/sofa.git
  cd sofa
  mkdir build && cd build
  
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DSOFA_ENABLE_GPU=ON \
    -DPYTHON_VERSION=3.10 \
    ..
  
  make -j8
  sudo make install
  ```

- [ ] **Task 2.2**: 简单软组织模型（肝脏）
  ```xml
  <!-- liver.scn -->
  <Node name="root" gravity="0 -9.81 0" dt="0.01">
      <!-- 求解器 -->
      <RequiredPlugin name="SofaOpenglVisual"/>
      <RequiredPlugin name="SofaBoundaryCondition"/>
      
      <VisualStyle displayFlags="showVisual showWireframe showBehaviorModels" />
      
      <Node name="Liver">
          <!-- 时间积分 -->
          <EulerImplicitSolver rayleighStiffness="0.1" rayleighMass="0.1" />
          <CGLinearSolver iterations="25" tolerance="1e-5" threshold="1e-5"/>
          
          <!-- 网格加载 -->
          <MeshGmshLoader name="loader" filename="liver.msh" />
          <TetrahedronSetTopologyContainer src="@loader" />
          <MechanicalObject name="mstate" src="@loader" />
          
          <!-- 质量 -->
          <DiagonalMass totalMass="1.5" />  <!-- 1.5 kg -->
          
          <!-- 有限元力场 (Neo-Hookean) -->
          <TetrahedronFEMForceField 
              name="FEM" 
              youngModulus="5000"    <!-- Pa -->
              poissonRatio="0.45"    <!-- 近不可压 -->
              method="large"         <!-- 大变形 -->
          />
          
          <!-- 固定约束（肝门静脉区域） -->
          <FixedConstraint indices="@loader.fixedPoints" />
          
          <!-- 可视化 -->
          <Node name="Visual">
              <OglModel name="visualModel" src="@../loader" color="0.8 0.3 0.2 1.0"/>
              <BarycentricMapping input="@../mstate" output="@visualModel" />
          </Node>
      </Node>
  </Node>
  ```

- [ ] **Task 2.3**: 器械-组织接触
  ```xml
  <!-- 在 liver.scn 中添加 -->
  <Node name="Tool">
      <MechanicalObject name="toolPos" position="0 0.05 0" />
      <SphereCollisionModel radius="0.005" />
      
      <!-- 碰撞响应 -->
      <CollisionResponse response="PenalityContactForceField" />
  </Node>
  
  <!-- 碰撞检测管道 -->
  <DefaultPipeline depth="6" />
  <BruteForceBroadPhase/>
  <BVHNarrowPhase/>
  <MinProximityIntersection alarmDistance="0.002" contactDistance="0.001"/>
  ```

- [ ] **Task 2.4**: Python 控制接口
  ```python
  import Sofa
  
  class LiverController(Sofa.Core.Controller):
      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.liver_node = None
          self.tool_node = None
      
      def onAnimateBeginEvent(self, event):
          # 每帧更新器械位置
          if self.tool_node:
              tool_pos = self.get_robot_tool_pos()  # 从 MuJoCo 获取
              self.tool_node.getObject('toolPos').position = tool_pos
      
      def get_contact_forces(self):
          # 提取接触力
          mstate = self.liver_node.getObject('mstate')
          forces = mstate.force.value
          return forces
  ```

#### 验收标准
- ✅ 肝脏模型稳定变形
- ✅ 器械接触产生真实变形
- ✅ 接触力可提取
- ✅ 运行速度 > 20 FPS

---

### Sprint 3: 混合架构原型（Week 9-12）

#### 任务清单
- [ ] **Task 3.1**: MuJoCo-SOFA 接口层
  ```python
  class HybridSimulator:
      def __init__(self):
          self.mujoco_sim = SurgicalRobotSim("surgical_robot.xml")
          self.sofa_sim = Sofa.Simulation.Node("root")
          # 加载 SOFA 场景
          Sofa.Simulation.load(self.sofa_sim, "liver.scn")
          Sofa.Simulation.init(self.sofa_sim)
          
          self.dt = 0.01  # 10 ms 时间步
      
      def step(self, robot_action):
          # 1. MuJoCo 更新机器人
          robot_state = self.mujoco_sim.step(robot_action)
          
          # 2. 提取器械位置
          tool_pos = robot_state['tool_pos']
          tool_vel = np.diff(tool_pos) / self.dt
          
          # 3. 传递给 SOFA
          self.update_tool_in_sofa(tool_pos)
          
          # 4. SOFA 更新软组织
          Sofa.Simulation.animate(self.sofa_sim, self.dt)
          
          # 5. 提取接触力
          contact_force = self.get_sofa_contact_force()
          
          # 6. 反馈给 MuJoCo
          self.mujoco_sim.data.xfrc_applied[tool_body_id, :3] = contact_force
          
          return robot_state, self.get_tissue_state()
      
      def update_tool_in_sofa(self, pos):
          tool_node = self.sofa_sim.getChild('Tool')
          tool_mstate = tool_node.getObject('toolPos')
          tool_mstate.position.value = pos
      
      def get_sofa_contact_force(self):
          # 从 SOFA 碰撞模型提取力
          collision_model = self.sofa_sim.getChild('Liver').getChild('Tool')
          contact_force = collision_model.getContactForce()
          return contact_force
  ```

- [ ] **Task 3.2**: 时间同步与稳定性
  ```python
  def synchronized_step(self, robot_action):
      # 子步法：SOFA 需要更小时间步
      mujoco_dt = 0.002  # 2 ms
      sofa_dt = 0.001    # 1 ms
      
      for _ in range(int(sofa_dt / mujoco_dt)):
          self.mujoco_sim.step(robot_action)
          
          # 每个 MuJoCo 步后更新 SOFA 两次
          tool_pos = self.mujoco_sim.get_tool_pos()
          self.update_tool_in_sofa(tool_pos)
          Sofa.Simulation.animate(self.sofa_sim, sofa_dt)
  ```

- [ ] **Task 3.3**: 第一个医疗场景：穿刺定位
  ```python
  class NeedleInsertionTask:
      def __init__(self):
          self.hybrid_sim = HybridSimulator()
          self.target_pos = np.array([0.02, 0.03, 0.05])  # 肿瘤位置
      
      def reset(self):
          self.hybrid_sim.reset()
          # 随机化目标位置（域随机化）
          self.target_pos += np.random.randn(3) * 0.01
      
      def step(self, action):
          # 执行穿刺动作
          robot_state, tissue_state = self.hybrid_sim.step(action)
          
          # 计算奖励
          needle_tip = robot_state['tool_pos']
          distance = np.linalg.norm(needle_tip - self.target_pos)
          
          reward = -distance * 100  # 距离奖励
          
          # 惩罚过大力
          contact_force = self.hybrid_sim.get_contact_force()
          if np.linalg.norm(contact_force) > 5.0:  # 5 N
              reward -= 50
          
          # 成功判断
          done = distance < 0.003  # 3 mm 以内
          
          return obs, reward, done, {}
  ```

- [ ] **Task 3.4**: Demo 视频录制
  ```python
  import cv2
  
  def record_demo():
      sim = HybridSimulator()
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      video = cv2.VideoWriter('demo.mp4', fourcc, 30, (1920, 1080))
      
      for i in range(300):  # 10 秒 @ 30 FPS
          # 执行随机动作
          action = np.random.randn(4) * 0.1
          sim.step(action)
          
          # 渲染
          frame = sim.render()
          video.write(frame)
      
      video.release()
  ```

#### 验收标准
- ✅ MuJoCo 和 SOFA 双向耦合稳定
- ✅ 器械-组织交互真实感
- ✅ Demo 视频可展示
- ✅ 帧率 > 15 FPS

#### 里程碑输出
📹 **Demo 视频**: 手术机器人穿刺肝脏肿瘤（30秒）

---

## 🚀 阶段2: 核心集成（Month 4-6）

### Sprint 4: Dynami-CAL 软组织扩展（Week 13-16）

#### 任务清单
- [ ] **Task 4.1**: 训练数据生成
  ```python
  def generate_training_data():
      """从 SOFA 仿真生成 Dynami-CAL 训练数据"""
      sofa_sim = load_sofa_liver()
      
      dataset = []
      for episode in range(1000):
          # 随机初始化
          sofa_sim.reset()
          apply_random_deformation()
          
          for step in range(100):
              # 记录状态
              pos = sofa_sim.get_node_positions()
              vel = sofa_sim.get_node_velocities()
              forces = sofa_sim.get_node_forces()
              
              # 构建图
              edge_index = radius_graph(pos, r=0.02)
              
              data = Data(
                  pos=torch.tensor(pos),
                  vel=torch.tensor(vel),
                  force=torch.tensor(forces),
                  edge_index=edge_index
              )
              dataset.append(data)
              
              # 更新
              sofa_sim.step()
      
      return dataset
  ```

- [ ] **Task 4.2**: 软组织本构模型
  ```python
  class SoftTissueGNN(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.gnn = DynamiCAL(
              hidden_dim=128,
              num_layers=5,
              output_dim=3  # 力向量
          )
          
          # 超弹性参数预测器
          self.material_encoder = MLP([16, 64, 64, 2])  # [E, ν]
      
      def forward(self, data):
          # 1. 构建边坐标系
          edge_frames, dist = compute_edge_frames(
              data.pos, data.vel, data.ang_vel, data.edge_index
          )
          
          # 2. 计算变形梯度（软组织特有）
          F = self.compute_deformation_gradient(data.pos, edge_index)
          J = torch.det(F)  # 体积比
          
          # 3. 超弹性应力
          E, nu = self.material_encoder(data.node_features)
          stress = self.neo_hookean_stress(F, E, nu)
          
          # 4. 内力
          internal_forces = self.stress_to_force(stress, edge_frames)
          
          # 5. GNN 学习残差
          residual = self.gnn(data)
          
          return internal_forces + residual
      
      def compute_deformation_gradient(self, pos, edge_index):
          """计算每个单元的变形梯度"""
          # 简化：用邻居位移估计梯度
          row, col = edge_index
          rel_pos = pos[row] - pos[col]
          
          # 用最小二乘拟合梯度
          # F = Σ(Δx ⊗ Δx₀⁻¹)
          pass
      
      def neo_hookean_stress(self, F, E, nu):
          """Neo-Hookean 本构模型"""
          J = torch.det(F)
          F_inv_T = torch.inverse(F).transpose(-1, -2)
          
          # Lamé 参数
          mu = E / (2 * (1 + nu))
          lam = E * nu / ((1 + nu) * (1 - 2 * nu))
          
          # Cauchy 应力
          sigma = mu * (F - F_inv_T) + lam * torch.log(J) * F_inv_T
          return sigma
  ```

- [ ] **Task 4.3**: 训练与验证
  ```python
  def train_soft_tissue_gnn():
      model = SoftTissueGNN()
      optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
      
      dataset = generate_training_data()
      train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
      
      for epoch in range(100):
          for batch in train_loader:
              optimizer.zero_grad()
              
              # 预测
              pred_forces = model(batch)
              
              # 损失
              loss_force = F.mse_loss(pred_forces, batch.force)
              
              # 物理约束（守恒性）
              loss_momentum = torch.sum(pred_forces, dim=0).pow(2).mean()
              
              loss = loss_force + 0.1 * loss_momentum
              
              loss.backward()
              optimizer.step()
          
          # 验证
          val_error = validate_on_sofa()
          print(f"Epoch {epoch}, Val Error: {val_error:.4f} mm")
  ```

#### 验收标准
- ✅ GNN 变形误差 < 5mm
- ✅ 推理速度 > 100 FPS
- ✅ 守恒性误差 < 1%

---

### Sprint 5: 流体模拟（SPH）（Week 17-20）

#### 任务清单
- [ ] **Task 5.1**: SPH 核心实现
  ```python
  class SPH_Simulator:
      def __init__(self, num_particles=50000):
          self.pos = torch.randn(num_particles, 3) * 0.1
          self.vel = torch.zeros(num_particles, 3)
          self.density = torch.ones(num_particles) * 1060  # kg/m³
          
          self.h = 0.01  # 光滑长度 (m)
          self.dt = 0.0001  # 0.1 ms
      
      def compute_density(self):
          # 邻域搜索
          edge_index = radius_graph(self.pos, r=2*self.h)
          row, col = edge_index
          
          # Poly6 核
          r = torch.norm(self.pos[row] - self.pos[col], dim=-1)
          W = (315 / (64 * np.pi * self.h**9)) * (self.h**2 - r**2)**3
          W = W.clamp(min=0)
          
          # 密度累加
          self.density = scatter_add(W, row, dim=0, dim_size=len(self.pos))
      
      def compute_pressure_forces(self):
          # 状态方程
          p = 1000 * ((self.density / 1060)**7 - 1)  # Tait 方程
          
          edge_index = radius_graph(self.pos, r=2*self.h)
          row, col = edge_index
          
          # Spiky 核梯度
          r_ij = self.pos[row] - self.pos[col]
          r = torch.norm(r_ij, dim=-1, keepdim=True) + 1e-8
          grad_W = -(45 / (np.pi * self.h**6)) * (self.h - r)**2 * r_ij / r
          
          # 压力力
          f_pressure = -scatter_add(
              (p[row] + p[col]) / (2 * self.density[col]) * grad_W,
              row, dim=0, dim_size=len(self.pos)
          )
          
          return f_pressure
      
      def step(self):
          self.compute_density()
          
          f_pressure = self.compute_pressure_forces()
          f_viscosity = self.compute_viscosity_forces()
          f_gravity = torch.tensor([0, -9.81, 0]) * self.density
          
          # 更新
          f_total = f_pressure + f_viscosity + f_gravity
          self.vel += f_total / self.density.unsqueeze(-1) * self.dt
          self.pos += self.vel * self.dt
          
          # 边界条件
          self.enforce_boundaries()
  ```

- [ ] **Task 5.2**: 流固耦合
  ```python
  class FluidSolidCoupling:
      def __init__(self, fluid_sim, solid_sim):
          self.fluid = fluid_sim
          self.solid = solid_sim
      
      def step(self):
          # 1. 检测流体粒子-固体接触
          solid_pos = self.solid.get_surface_nodes()
          fluid_pos = self.fluid.pos
          
          distances, indices = knn(fluid_pos, solid_pos, k=5)
          
          # 2. 固体速度 → 流体边界条件
          for i, idx in enumerate(indices):
              if distances[i] < 0.01:  # 1 cm 耦合距离
                  self.fluid.vel[i] = self.solid.get_velocity(idx)
          
          # 3. 流体压力 → 固体表面力
          pressure_forces = self.compute_fluid_pressure_on_solid()
          self.solid.apply_surface_forces(pressure_forces)
  ```

#### 验收标准
- ✅ 血液流动视觉真实
- ✅ 质量守恒误差 < 1%
- ✅ 50K 粒子 @ 30 FPS

---

### Sprint 6: 多模态感知（Week 21-24）

#### 任务清单
- [ ] **Task 6.1**: 超声波模拟器
  ```python
  class UltrasoundSimulator:
      def __init__(self, tissue_model):
          self.tissue = tissue_model
          self.probe_freq = 5e6  # 5 MHz
          self.speed_of_sound = 1540  # m/s
      
      def render(self, probe_pose):
          # 1. 射线追踪
          rays = self.generate_ultrasound_beams(probe_pose, num_rays=256)
          
          image = np.zeros((512, 512))
          
          for i, ray in enumerate(rays):
              # 2. 与组织交互
              reflections = []
              attenuation = 1.0
              
              for intersection in self.tissue.intersect(ray):
                  # 声阻抗不匹配 → 反射
                  Z1 = self.tissue.get_impedance(intersection.prev_material)
                  Z2 = self.tissue.get_impedance(intersection.next_material)
                  R = ((Z2 - Z1) / (Z2 + Z1))**2  # 反射系数
                  
                  # 衰减 (Beer-Lambert)
                  distance = intersection.distance
                  attenuation *= np.exp(-0.5 * self.probe_freq * distance / 1e6)
                  
                  intensity = R * attenuation
                  reflections.append((intersection.depth, intensity))
              
              # 3. 波束成形
              for depth, intensity in reflections:
                  row = int(depth / 0.1 * 512)  # 10 cm 深度
                  image[row, i] = intensity
          
          # 4. 添加噪声和伪影
          image = self.add_speckle_noise(image, snr=20)
          image = self.add_acoustic_shadow(image)
          
          return image
  ```

- [ ] **Task 6.2**: MRI 数据集成
  ```python
  import SimpleITK as sitk
  
  def load_patient_mri(dicom_dir):
      # 读取 DICOM 序列
      reader = sitk.ImageSeriesReader()
      series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
      dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
      reader.SetFileNames(dicom_files)
      image = reader.Execute()
      
      # 转为 NumPy
      array = sitk.GetArrayFromImage(image)
      spacing = image.GetSpacing()
      origin = image.GetOrigin()
      
      return array, spacing, origin
  
  def segment_liver(mri_array):
      # U-Net 分割
      model = torch.load('liver_segmentation_model.pth')
      mask = model(mri_array)
      return mask
  
  def generate_fem_mesh(mask, spacing):
      # 体素→四面体网格
      import pymesh
      
      # Marching cubes
      vertices, faces = measure.marching_cubes(mask, level=0.5)
      vertices *= spacing  # 真实物理尺寸
      
      # 四面体化
      mesh = pymesh.form_mesh(vertices, faces)
      mesh, _ = pymesh.tetrahedralize(mesh, order=2, max_tet_volume=0.001)
      
      return mesh
  ```

#### 验收标准
- ✅ 超声图像 SSIM > 0.75
- ✅ MRI 分割 Dice > 0.90
- ✅ 网格生成自动化

---

## 🏆 阶段3: 临床优化（Month 7-9）

### Sprint 7-9: 详细计划（简略）

**Sprint 7**: 强化学习训练框架  
**Sprint 8**: 迁移学习与域随机化  
**Sprint 9**: 临床验证与用户测试

（详细任务省略，见完整路线图）

---

## 📊 资源规划

### 人员配置
- **核心开发**: 1人（Taisen）
- **咨询顾问**: 外科医生 1人（兼职）
- **GPU 资源**: NVIDIA A100 (40GB) × 1

### 时间分配
| 阶段 | 核心开发 | 测试 | 文档 | 总计 |
|------|---------|------|------|------|
| 阶段1 | 60h | 20h | 10h | 90h |
| 阶段2 | 80h | 30h | 15h | 125h |
| 阶段3 | 70h | 40h | 20h | 130h |
| **总计** | 210h | 90h | 45h | **345h** |

### 预算估算
```yaml
硬件:
  GPU 租用 (9个月): $3000
  存储 (2TB): $200
  
软件:
  SOFA 许可: 开源免费
  MuJoCo: 开源免费
  
数据:
  医疗图像数据集: $500 (公开数据集)
  
总计: ~$3700
```

---

## ⚠️ 风险管理

| 风险 | 概率 | 影响 | 缓解策略 | 应急计划 |
|------|------|------|----------|----------|
| SOFA-MuJoCo 耦合不稳定 | 高 | 高 | 提前原型验证 | 降级为纯 SOFA |
| 实时性不达标 | 中 | 高 | GPU 性能分析 | 降低仿真分辨率 |
| 医疗数据获取困难 | 中 | 中 | 使用公开数据集 | 合成数据 |
| 软组织模型过于简化 | 低 | 中 | 文献调研 | 专家咨询 |

---

## ✅ 验收标准（总体）

### 技术指标
- [ ] 帧率 ≥ 30 FPS
- [ ] 软组织变形误差 < 5mm
- [ ] 接触力误差 < 1N
- [ ] 超声图像 SSIM > 0.75
- [ ] RL 训练收敛（穿刺任务）

### 可演示场景
- [ ] 肝脏穿刺活检
- [ ] 软组织抓取
- [ ] 出血控制

### 文档交付
- [ ] 用户手册
- [ ] API 文档
- [ ] 技术报告

---

## 📝 下一步行动

1. **本周**: 完成 Sprint 1 Task 1.1-1.2
2. **审查点**: 每周五技术评审
3. **里程碑**: Month 3 Demo 视频

**开始日期**: 2026年2月5日  
**首次评审**: 2026年2月14日
