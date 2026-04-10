# IGZO 迁移率分析中当前采用的散射机制与公式

这份文档只讨论当前代码中“迁移率分析”实际使用的散射机制、物理公式和参数，不涉及 MC 飞行、边界、泊松或其他过程。

对应代码入口：

- 迁移率积分：[scripts/analyze_scattering_mobility.py](/home/ic/3dmc_Si_ylx_mod/3DMCpy/scripts/analyze_scattering_mobility.py)
- 散射率构建：[src/physics/band_structure.py](/home/ic/3dmc_Si_ylx_mod/3DMCpy/src/physics/band_structure.py)

## 1. 当前迁移率分析实际包含的散射机制

当前 IGZO 迁移率分析只包含声子散射，不包含以下机制：

- 杂质散射：未并入当前迁移率积分
- 表面散射：未并入当前迁移率积分
- 电子-电子散射：未并入当前迁移率积分

当前真正参与总散射率的 5 个分支是：

1. 声学散射 `acoustic`
2. LO 光学声子吸收 `lo_abs`
3. LO 光学声子发射 `lo_ems`
4. TO 光学声子吸收 `to_abs`
5. TO 光学声子发射 `to_ems`

总散射率写成：

\[
\Gamma_{\mathrm{tot}}(E)=
\Gamma_{\mathrm{ac}}(E)+
\Gamma_{\mathrm{LO,abs}}(E)+
\Gamma_{\mathrm{LO,ems}}(E)+
\Gamma_{\mathrm{TO,abs}}(E)+
\Gamma_{\mathrm{TO,ems}}(E)
\]

相应的弛豫时间为：

\[
\tau(E)=\frac{1}{\Gamma_{\mathrm{tot}}(E)}
\]

## 2. 当前迁移率公式

当前脚本使用各向同性抛物线近似：

\[
m_d = (m_t^2 m_l)^{1/3}
\]

\[
v^2(E)=\frac{2 E q}{m_d}
\]

由于采用各向同性平均，

\[
\langle v_x^2(E)\rangle = \frac{v^2(E)}{3}
\]

电子浓度写成：

\[
n=\int g(E)f_0(E)\,dE
\]

迁移率写成：

\[
\mu = \frac{1}{n}\int \frac{v^2(E)}{3}\,\tau(E)\,
\left(-\frac{\partial f_0}{\partial E}\right)\,g(E)\,dE
\]

其中：

- \(g(E)\) 是输入 DOS
- \(f_0(E)\) 是费米分布
- \(\tau(E)\) 由当前总声子散射率决定

费米分布为：

\[
f_0(E)=\frac{1}{1+\exp\left(\frac{E-E_F}{k_B T}\right)}
\]

## 3. 色散与动量关系

当前散射率代码采用等效质量近似，并允许非抛物性参数 \(\alpha\)：

\[
k(E)=\frac{\sqrt{2m_d\,E\,(1+\alpha E)\,q}}{\hbar}
\]

当前你的设置中：

\[
\alpha = 0
\]

因此实际上退化为普通抛物线：

\[
k(E)=\frac{\sqrt{2m_d E q}}{\hbar}
\]

## 4. 声学散射公式

当前代码里，声学散射最终只输出一个总分支 `acoustic`，但这个总分支本身是由四项相加得到的：

\[
\Gamma_{\mathrm{ac}}(E)=
\Gamma_{\mathrm{LA,abs}}(E)+
\Gamma_{\mathrm{LA,ems}}(E)+
\Gamma_{\mathrm{TA,abs}}(E)+
\Gamma_{\mathrm{TA,ems}}(E)
\]

### 4.1 LA 吸收

\[
\Gamma_{\mathrm{LA,abs}}(E)
=
\frac{m_d D_{\mathrm{LA}}^2}{4\pi \rho \hbar^2 k(E)}
\int dq\,
\frac{|I(q)|^2 q^3}{\omega_{\mathrm{LA}}(q)}
N_{\mathrm{LA}}(q)\,
\Theta_{\mathrm{LA,abs}}(E,q)
\]

### 4.2 LA 发射

\[
\Gamma_{\mathrm{LA,ems}}(E)
=
\frac{m_d D_{\mathrm{LA}}^2}{4\pi \rho \hbar^2 k(E)}
\int dq\,
\frac{|I(q)|^2 q^3}{\omega_{\mathrm{LA}}(q)}
\bigl(N_{\mathrm{LA}}(q)+1\bigr)\,
\Theta_{\mathrm{LA,ems}}(E,q)
\]

并且发射必须满足末态能量为正，也就是：

\[
E > \hbar\omega_{\mathrm{LA}}(q)
\]

### 4.3 TA 吸收

\[
\Gamma_{\mathrm{TA,abs}}(E)
=
\frac{m_d D_{\mathrm{TA}}^2}{4\pi \rho \hbar^2 k(E)}
\int dq\,
\frac{|I(q)|^2 q^3}{\omega_{\mathrm{TA}}(q)}
N_{\mathrm{TA}}(q)\,
\Theta_{\mathrm{TA,abs}}(E,q)
\]

### 4.4 TA 发射

\[
\Gamma_{\mathrm{TA,ems}}(E)
=
\frac{m_d D_{\mathrm{TA}}^2}{4\pi \rho \hbar^2 k(E)}
\int dq\,
\frac{|I(q)|^2 q^3}{\omega_{\mathrm{TA}}(q)}
\bigl(N_{\mathrm{TA}}(q)+1\bigr)\,
\Theta_{\mathrm{TA,ems}}(E,q)
\]

同样需要满足：

\[
E > \hbar\omega_{\mathrm{TA}}(q)
\]

### 4.5 代码中的合并写法

在当前代码实现里，上面四项被合并成：

\[
\Gamma_{\mathrm{ac}}(E)
=
\frac{m_d}{4\pi \rho \hbar^2 k(E)}
\int dq\,
\left[
D_{\mathrm{LA}}^2\,W_{\mathrm{LA}}(E,q)
+
D_{\mathrm{TA}}^2\,W_{\mathrm{TA}}(E,q)
\right]
\]

其中

\[
W_{\mathrm{LA}}(E,q)=
\frac{|I(q)|^2 q^3}{\omega_{\mathrm{LA}}(q)}
\left[
N_{\mathrm{LA}}(q)\,\Theta_{\mathrm{LA,abs}}(E,q)
+
\bigl(N_{\mathrm{LA}}(q)+1\bigr)\,\Theta_{\mathrm{LA,ems}}(E,q)
\right]
\]

\[
W_{\mathrm{TA}}(E,q)=
\frac{|I(q)|^2 q^3}{\omega_{\mathrm{TA}}(q)}
\left[
N_{\mathrm{TA}}(q)\,\Theta_{\mathrm{TA,abs}}(E,q)
+
\bigl(N_{\mathrm{TA}}(q)+1\bigr)\,\Theta_{\mathrm{TA,ems}}(E,q)
\right]
\]

这里：

- \(\rho\) 是材料质量密度
- \(D_{\mathrm{LA}},D_{\mathrm{TA}}\) 是声学形变势
- \(\omega_{\mathrm{LA}}(q),\omega_{\mathrm{TA}}(q)\) 来自声子色散表
- \(N(q)\) 是玻色分布
- \(\Theta_{\mathrm{LA,abs}}, \Theta_{\mathrm{LA,ems}}, \Theta_{\mathrm{TA,abs}}, \Theta_{\mathrm{TA,ems}}\) 都表示对应过程的动量和能量守恒条件
- \(I(q)\) 是形状因子

## 5. LO 光学声子吸收/发射

### LO 吸收

\[
\Gamma_{\mathrm{LO,abs}}(E)
=
\frac{m_d D_{\mathrm{LO}}^2}{4\pi \rho \hbar^2 k(E)}
\int dq\,
\frac{|I(q)|^2 q^2}{\omega_{\mathrm{LO}}(q)}
N_{\mathrm{LO}}(q)\,\Theta_{\mathrm{LO,abs}}
\]

### LO 发射

\[
\Gamma_{\mathrm{LO,ems}}(E)
=
\frac{m_d D_{\mathrm{LO}}^2}{4\pi \rho \hbar^2 k(E)}
\int dq\,
\frac{|I(q)|^2 q^2}{\omega_{\mathrm{LO}}(q)}
\bigl(N_{\mathrm{LO}}(q)+1\bigr)\,\Theta_{\mathrm{LO,ems}}
\]

并且发射还要满足：

\[
E > \hbar\omega_{\mathrm{LO}}(q)
\]

## 6. TO 光学声子吸收/发射

### TO 吸收

\[
\Gamma_{\mathrm{TO,abs}}(E)
=
\frac{m_d D_{\mathrm{TO}}^2}{4\pi \rho \hbar^2 k(E)}
\int dq\,
\frac{|I(q)|^2 q^2}{\omega_{\mathrm{TO}}(q)}
N_{\mathrm{TO}}(q)\,\Theta_{\mathrm{TO,abs}}
\]

### TO 发射

\[
\Gamma_{\mathrm{TO,ems}}(E)
=
\frac{m_d D_{\mathrm{TO}}^2}{4\pi \rho \hbar^2 k(E)}
\int dq\,
\frac{|I(q)|^2 q^2}{\omega_{\mathrm{TO}}(q)}
\bigl(N_{\mathrm{TO}}(q)+1\bigr)\,\Theta_{\mathrm{TO,ems}}
\]

同样要求：

\[
E > \hbar\omega_{\mathrm{TO}}(q)
\]

## 7. 声子占据数

所有声子支都用玻色分布：

\[
N_\lambda(q)=
\frac{1}{\exp\left(\frac{\hbar\omega_\lambda(q)}{k_B T}\right)-1}
\]

其中 \(\lambda\) 可以是 LA、TA、LO、TO。

## 8. 无序增强因子

当前总散射率还会统一乘一个无序增强因子：

\[
\Gamma_i^{\mathrm{final}}(E)=S_{\mathrm{dis}}(E)\,\Gamma_i(E)
\]

当前启用的模型是：

`linear_tail_enhancement`

公式为：

\[
S_{\mathrm{dis}}(E)=\exp\left(\frac{\Delta E(E)}{k_B T}\right)
\]

其中：

\[
\Delta E(E)=
E_{\mathrm{tail}}
\left(1-\frac{E}{E_{\mathrm{cutoff}}}\right),
\qquad E<E_{\mathrm{cutoff}}
\]

当 \(E\ge E_{\mathrm{cutoff}}\) 时：

\[
\Delta E(E)=0,\qquad S_{\mathrm{dis}}(E)=1
\]

## 9. 当前 IGZO 参数值

以下是你当前输入和代码实际生效的参数。

### 9.1 材料与有效质量

- 材料：IGZO
- 质量密度：

\[
\rho = 6.10\times 10^3\ \mathrm{kg/m^3}
\]

- 纵向有效质量：

\[
m_l = 0.268\,m_0
\]

- 横向有效质量：

\[
m_t = 0.254\,m_0
\]

- 迁移率积分使用的各向同性等效质量：

\[
m_d = (m_t^2 m_l)^{1/3} = 0.258583460088\,m_0
\]

### 9.2 当前开启的散射分支

当前 `input.txt` 中：

- `acoustic`
- `lo_abs`
- `lo_ems`
- `to_abs`
- `to_ems`

即 5 个声子分支全部开启。

### 9.3 模型名

- 声学散射模型：`deformation_potential_acoustic`
- LO 光学散射模型：`deformation_potential_optical`
- TO 光学散射模型：`deformation_potential_optical`
- 无序模型：`linear_tail_enhancement`

### 9.4 当前数值参数

- 声学形变势：

\[
D_{\mathrm{LA}} = D_{\mathrm{TA}} = 5.0\ \mathrm{eV}
\]

- LO 光学形变势：

\[
D_{\mathrm{LO}} = 5.0\times 10^5\ \mathrm{eV/m}
\]

- TO 光学形变势：

\[
D_{\mathrm{TO}} = 5.0\times 10^5\ \mathrm{eV/m}
\]

- 非抛物性参数：

\[
\alpha = 0\ \mathrm{eV^{-1}}
\]

- 无序尾态能量尺度：

\[
E_{\mathrm{tail}} = 0.18\ \mathrm{eV}
\]

- 无序增强截止能量：

\[
E_{\mathrm{cutoff}} = 10.0\ \mathrm{eV}
\]

### 9.5 迁移率积分中当前费米能级

当前 `ldg.txt` 中设置为：

\[
E_F = 3.13\ \mathrm{eV}
\]

当前导带参考能量在分析脚本里采用：

\[
E_c = 3.33\ \mathrm{eV}
\]

因此迁移率积分使用的相对费米能级为：

\[
E_F - E_c = -0.20\ \mathrm{eV}
\]

## 10. 当前应如何理解

当前这套迁移率分析，实质上是：

1. 用 IGZO 声子色散表数值构建
   \(\Gamma_{\mathrm{tot}}(E,T)\)
2. 用 DOS 和费米分布构建
   \(n\) 与 \(-\partial f_0/\partial E\)
3. 用各向同性抛物线近似
   \(\mu(T)\)

因此目前的迁移率分析是：

- 包含声学与光学声子散射
- 包含无序增强因子
- 不包含杂质散射与表面散射
- 速度采用各向同性抛物线近似，而不是直接用全带速度
