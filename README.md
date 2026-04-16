## 3DMCpy 脚本说明

散射模型配置说明见：
[docs/scattering_models.md](/home/ic/3dmc_Si_ylx_mod/3DMCpy/docs/scattering_models.md)

## 主程序输入补充

### 输出目录命名

主程序默认会在 `output_dir` 下按时间戳新建输出目录。  
如果希望直接指定输出目录名，可以在 `input.txt` 中加入：

```txt
output_dir = ./output
output_name = my_run_name
```

这样输出会写到：

```txt
./output/my_run_name
```

说明：
- 如果同时设置了 `ResumeFromOutputDir`，则续跑仍然直接写回原输出目录，`output_name` 不生效。
- `output_name` 只决定目录名，不会改动其他输入参数。

### `input_extract_mobility` 批量电势初始化示例

基于 [input_extract_mobility/input.txt](/home/ic/3dmc_Si_ylx_mod/3DMCpy/input_extract_mobility/input.txt)，仓库中已生成以下输入：

- [input_vd0.1.txt](/home/ic/3dmc_Si_ylx_mod/3DMCpy/input_extract_mobility/input_vd0.1.txt)
- [input_vd0.25.txt](/home/ic/3dmc_Si_ylx_mod/3DMCpy/input_extract_mobility/input_vd0.25.txt)
- [input_vd0.5.txt](/home/ic/3dmc_Si_ylx_mod/3DMCpy/input_extract_mobility/input_vd0.5.txt)
- [input_vd1.txt](/home/ic/3dmc_Si_ylx_mod/3DMCpy/input_extract_mobility/input_vd1.txt)
- [input_vd2.txt](/home/ic/3dmc_Si_ylx_mod/3DMCpy/input_extract_mobility/input_vd2.txt)
- [input_vd4.txt](/home/ic/3dmc_Si_ylx_mod/3DMCpy/input_extract_mobility/input_vd4.txt)

这些输入相对模板只改了：
- `InitialPotentialFile`
- `output_name`

批量运行脚本：
- [run_all_vd_potential_cases.sh](/home/ic/3dmc_Si_ylx_mod/3DMCpy/input_extract_mobility/run_all_vd_potential_cases.sh)

用法：

```bash
cd /home/ic/3dmc_Si_ylx_mod/3DMCpy
PYTHON_BIN=/home/ic/miniconda3/envs/3dmc/bin/python \
  ./input_extract_mobility/run_all_vd_potential_cases.sh
```

脚本会按以下顺序运行：

```txt
vd0.1 -> vd0.25 -> vd0.5 -> vd1 -> vd2 -> vd4
```

下面只保留当前仍然可用、且职责明确的脚本。重复功能已合并：

- `plot_energy_cdf.py` 已删除  
  原因：其功能已被 [plot_energy_cdf_compare.py](/home/ic/3dmc_Si_ylx_mod/3DMCpy/scripts/plot_energy_cdf_compare.py) 完全覆盖，后者既支持单文件，也支持多文件对比。

### 1. 散射率与迁移率分析

#### `scripts/analyze_scattering_mobility.py`
用途：
- 在指定温度下重建 IGZO 散射率
- 导出各分支散射率、总散射率
- 用 DOS + 抛物线近似速度估算迁移率

基本用法：
```bash
python ./scripts/analyze_scattering_mobility.py \
  --input ./input/input.txt \
  --temperature 300
```

常用参数：
- `--input`：主输入文件
- `--temperature`：温度，单位 `K`
- `--outdir`：输出目录
- `--dos`：DOS 文件路径覆盖
- `--ef`：费米能级覆盖，单位 `eV`
- `--ec-ref`：导带参考能级覆盖，单位 `eV`
- `--ml`、`--mt`：有效质量覆盖

输出：
- `scattering_rates_vs_energy.csv`
- `scattering_rates_log.png`
- `scattering_rates_linear.png`
- `mobility_integrand.csv`
- `mobility_integrands.png`
- `mobility_summary.txt`

#### `scripts/analyze_mobility_vs_temperature.py`
用途：
- 扫描温度范围
- 输出迁移率随温度变化的 `csv` 和图片
- 支持直接指定浓度
- 支持按温度点并行加速

基本用法：
```bash
python ./scripts/analyze_mobility_vs_temperature.py \
  --input ./input/input.txt \
  --tmin 250 \
  --tmax 350 \
  --tstep 1
```

常用参数：
- `--input`：主输入文件
- `--tmin`、`--tmax`、`--tstep`：温度范围，单位 `K`
- `--outdir`：输出目录
- `--dos`、`--ef`、`--ec-ref`、`--ml`、`--mt`：同单温脚本
- `--n-cm3`：直接指定载流子浓度，单位 `cm^-3`
- `--workers`：温度点并行进程数
  - `--workers 0`：自动，当前默认
  - `--workers 1`：串行
  - `--workers N`：指定 `N` 个进程

示例：
```bash
python ./scripts/analyze_mobility_vs_temperature.py \
  --input ./input/input.txt \
  --tmin 250 \
  --tmax 350 \
  --tstep 1 \
  --n-cm3 1e17 \
  --workers 4
```

输出：
- `mobility_vs_temperature.csv`
- `mobility_vs_temperature.png`
- `mobility_vs_temperature_summary.txt`

#### `scripts/analyze_total_scattering_vs_temperature.py`
用途：
- 在多个指定温度下重建总散射率
- 对比总散射率随能量的变化

基本用法：
```bash
python ./scripts/analyze_total_scattering_vs_temperature.py \
  --input ./input/input.txt \
  --temperatures 300 350 400
```

输出：
- `total_scattering_vs_energy.csv`
- `total_scattering_vs_energy.png`
- `total_scattering_vs_energy_summary.txt`

### 2. 运行结果快速可视化

#### `scripts/plot_scattering_rates.py`
用途：
- 从一次 MC 运行输出目录中的 `Scatter/scattering_rates.txt` 快速画图

基本用法：
```bash
python ./scripts/plot_scattering_rates.py ./output/20260309_125350
```

说明：
- 这是针对“已有运行结果”的快速可视化脚本
- 如果你要“按输入参数重建散射率”，优先用 `analyze_scattering_mobility.py`

#### `scripts/plot_monitor_current_compare.py`
用途：
- 将一个或多个运行目录中的监视面电流串接起来
- 在同一坐标系下画出多个监视面的电流

基本用法：
```bash
python ./scripts/plot_monitor_current_compare.py \
  --runs output/20260410_221407 output/20260411_132153
```

输出：
- `monitor_current_vs_step_compare.png`
- `monitor_current_vs_step_compare.csv`
- `monitor_current_vs_step_compare_summary.txt`

说明：
- `--runs` 传入顺序即连接顺序
- 适用于“续跑后分成多个输出目录”的电流拼接对比

### 3. 粒子分布分析

#### `scripts/plot_energy_cdf_compare.py`
用途：
- 画一个或多个粒子文件的能量 CDF
- 默认按 `abs(charge(/q))` 加权，也就是按真实粒子数等效统计

基本用法：
```bash
python ./scripts/plot_energy_cdf_compare.py \
  --series \
  "Initial::output/20260410_192152/Particles/initial_particles.txt" \
  "Final::output/20260410_192152/Snapshots/step_0007000/particles.txt" \
  --outdir output/20260410_192152/Particles
```

常用参数：
- `--series "标签::文件路径"`：可传多个
- `--outdir`：输出目录
- `--unweighted`：按超粒子条目数统计，而不是按真实粒子数等效统计

输出：
- `energy_cdf_compare.png`
- `energy_cdf_compare.csv`
- `energy_cdf_compare_summary.txt`

#### `scripts/analyze_nonequilibrium_distribution.py`
用途：
- 从 `particles.txt` 或 `initial_particles.txt` 提取非平衡能量分布
- 导出：
  - `p(E)`：概率密度
  - `n_E(E)`：占据态密度
  - `f_noneq(E) = n_E(E) / DOS(E)`

支持的输入：
- 某一步快照目录，例如 `Snapshots/step_0007000`
- 某一步的 `particles.txt`
- 初始粒子文件 `Particles/initial_particles.txt`
- 整个输出根目录，脚本会自动优先寻找初始粒子

基本用法：
```bash
python ./scripts/analyze_nonequilibrium_distribution.py \
  --input ./output/20260410_192152/Snapshots/step_0007000
```

初始粒子示例：
```bash
python ./scripts/analyze_nonequilibrium_distribution.py \
  --input ./output/20260410_192152/Particles/initial_particles.txt
```

说明：
- 统计按 `abs(charge(/q))` 加权，也就是按真实粒子数等效统计
- 对快照粒子，脚本会优先使用同一步的：
  - `electron_concentration_cells.txt`
  - `electron_concentration_nodes.txt`
  来恢复总体积，从而给出 `n_E(E)` 和 `f_noneq(E)`
- 对初始粒子，若快照体积文件不存在，脚本会回退到输出目录下的：
  - `Concentration/initial_electron_concentration_cells.txt`
  来恢复体积

输出：
- `nonequilibrium_distribution.csv`
- `nonequilibrium_distribution_summary.txt`
- `probability_density_vs_energy.png`
- `probability_density_vs_energy_log.png`
- `occupied_density_vs_energy.png`
- `occupied_density_vs_energy_log.png`
- `f_noneq_vs_energy.png`

#### `scripts/plot_velocity_cdf.py`
用途：
- 用初始粒子的 `k` 和给定有效质量，计算速度大小 CDF

基本用法：
```bash
python ./scripts/plot_velocity_cdf.py \
  --input ./output/20260309_125350 \
  --outdir ./output/20260309_125350/Particles \
  --mt 0.254 \
  --ml 0.268 \
  --a0 8.31e-10
```

#### `scripts/plot_particle_ev.py`
用途：
- 画粒子 `E-v` 散点，并与 `ml`、`mt` 两条解析曲线对比

基本用法：
```bash
python ./scripts/plot_particle_ev.py \
  --input ./output/20260309_125350/Particles/initial_particles.txt \
  --outdir ./output/20260309_125350/Particles \
  --mt 0.254 \
  --ml 0.268 \
  --a0 8.31e-10
```

### 4. 数据生成

#### `scripts/generate_dos_parabolic.py`
用途：
- 生成 3D 抛物线近似 DOS 表

基本用法：
```bash
python ./scripts/generate_dos_parabolic.py \
  --emax 8.0 \
  --de 0.001 \
  --ml 0.268 \
  --mt 0.254 \
  --out data/bands/DOS_parabolic.txt
```

#### `scripts/generate_bands_igzo.py`
用途：
- 按配置文件定义的 `k` 网格，生成 `bands_IGZO.txt`

基本用法：
```bash
python ./scripts/generate_bands_igzo.py --config ./scripts/bands_igzo_grid.txt
```

说明：
- 示例配置文件见 [bands_igzo_grid.txt](/home/ic/3dmc_Si_ylx_mod/3DMCpy/scripts/bands_igzo_grid.txt)

### 5. 推荐使用关系

如果目标是：

- 查看单温散射率和迁移率  
  用 `analyze_scattering_mobility.py`

- 查看迁移率-温度曲线  
  用 `analyze_mobility_vs_temperature.py`

- 查看总散射率在多个温度下的变化  
  用 `analyze_total_scattering_vs_temperature.py`

- 查看续跑前后或多次运行的电流变化  
  用 `plot_monitor_current_compare.py`

- 查看粒子能量分布对比  
  用 `plot_energy_cdf_compare.py`
