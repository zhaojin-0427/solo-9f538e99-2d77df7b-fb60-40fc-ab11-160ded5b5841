# 天文测光主平场制备服务

基于 **FastAPI + Astropy** 的主平场（master flat）制备接口。上传 3～12 张已扣偏置的
平场 FITS，服务按统一规则聚合，输出可直接用于平场校正的成品 FITS。

## 处理规则

1. **输入约束**
   - 帧数 3～12 张；每张恰好一幅**无压缩、二维、16 位整数**主图（`BITPIX=16`）。
   - 宽高均不超过 4096；同批帧尺寸必须一致。
   - 同批相机（`INSTRUME`）与滤镜（`FILTER`）标识必须一致（关键字缺失则不参与比对）。
   - 禁止压缩 HDU（`CompImageHDU`）与表 HDU。
2. **物理量换算**：保留原始整数读入，自行按 `BSCALE` / `BZERO` 换算
   （`value = raw * BSCALE + BZERO`），不依赖 Astropy 的自动缩放。
3. **有效像元**
   - 按 `BLANK` 关键字排除缺测像元；
   - 仅保留换算后 `0 < value < saturation`（严格大于零、严格低于饱和阈值）的像元；
   - 饱和阈值必须为正且有限。
4. **两阶段中位数聚合**
   - 每帧的有效像元除以**该帧有效像元全图中位数**做归一；
   - 对同一位置跨帧的有效归一值取**中位数**；
   - 某位置参与帧数少于 3 时输出 `NaN`。
5. **输出 FITS**
   - 主图（PRIMARY，`float32`）：校正系数；
   - 扩展图（`COUNTS`，`uint8`）：每个像元实际参与的帧数；
   - 头部记录帧数、饱和阈值、各帧中位数、输入清单与处理历史。
6. **整批失败语义**：文件截断、格式不符、任一帧无有效像元、写出失败均整批报错，
   成品先写同目录临时文件、回读校验通过后才原子改名，**只提供完整成品，不留半成品**；
   无论成功失败，请求结束即清理本批全部输入暂存。

## 磁盘暂存设计（紧凑工作盘）

工作盘只需放得下**一批**原始曝光，服务全程不复制输入：

- **每帧在盘上至多一份**：`POST /prepare` 用 python-multipart 推送式解析器逐块
  解析报文，文件字节直接落到该帧唯一暂存文件，**不经过
  SpooledTemporaryFile，也不产生任何上传解析临时文件**；每帧独立空目录隔离不可信数据。
- **单帧读取预算 = 2 × 帧像元数据量**：每帧只持有一个裸文件句柄，从 FITS 数据块
  偏移按需 `seek/read` 原始像元字节。结构校验只读头部卡片、不计像元读取；第一趟
  （直方图中位数）顺序读 1 倍，第二趟（行带聚合）再顺序读 1 倍，累计恰好 2 倍。
  命中操作系统页缓存的重复读取**同样计数**，超预算直接整批失败。
- **成品不与输入抢暂存盘的副本空间**：成品临时文件与成品同目录同卷，原子改名即可，
  无跨卷复制；输入暂存位于工作目录、成品位于输出目录。
- **清理语义**：成功或失败都 `rmtree` 本批暂存目录；写盘遇 ENOSPC 立即整批报错。

## 内存设计（256 MiB 容器预算）

最坏情形（12 帧 × 4096×4096）纯净子进程实测峰值约 **238 MiB**，余量约 18 MiB：

- 第一趟用 65536 桶的原始值**直方图**精确求每帧中位数，不缓存整帧有效值；
- 第二趟按**自适应行带（band）**聚合，行带高度由 `BAND_BUDGET_BYTES`（默认 8 MiB）
  反推；同一行带内各帧只按需读取该行带的像元字节，不重读其他区域；
- 帧轴中位数用**原地排序 + 索引选择**，避免 `np.nanmedian` 的整份条带复制与类型提升；
- 输出用 `ndarray.tofile` **流式直写** FITS 数据块，规避 `HDUList.writeto` 的大块内部缓冲；
- 服务串行处理（全局锁），避免并发制备叠加内存。

## 启动方式

### Docker Compose（推荐）

```bash
docker compose up --build
```

容器内存硬限制为 256 MiB（`mem_limit` / `memswap_limit`，swap 等额封顶防止换出扩容），
工作目录与成品目录共用命名卷 `flat_data`（挂载于 `/data`）。启动后访问：

- 交互文档（可直接上传）：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

### 本地直接运行

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口用法

`POST /prepare`，`multipart/form-data`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `files` | File[] | 3～12 张平场 FITS，单张上限 64 MiB |
| `saturation` | number | 正且有限的饱和阈值（与换算后数据同单位） |

成功返回 `application/fits` 成品附件；失败返回 `422`（输入/数据问题）或
`500`（磁盘/写出问题），响应体 `{"detail": "..."}`。

curl 示例：

```bash
curl -X POST http://localhost:8000/prepare \
  -F "saturation=4000" \
  -F "files=@flat_0.fits" \
  -F "files=@flat_1.fits" \
  -F "files=@flat_2.fits" \
  -o master_flat.fits
```

## 测试

```bash
pip install pytest
PYTHONPATH=. python -m pytest tests/ -q
```

- `tests/test_pipeline.py`、`tests/test_api.py`：单元/接口测试，含与 numpy 全量
  对照实现的数值比对、BLANK/BSCALE/BZERO、<3 帧 NaN、饱和/非正过滤、相机/滤镜与
  尺寸一致性、截断/格式/帧数/阈值等失败路径，以及**单帧 2 倍读取预算**、
  **暂存零复制/零临时文件**、失败清理等磁盘语义断言。
- `tests/scripts/stress_memory.py`：12 帧 4096² 端到端数值压测（含读取倍数输出）。
- `tests/scripts/measure_pipeline_only.py`：纯净子进程管线峰值内存测量。
- `tests/scripts/matrix_sizes.py`：多尺寸/帧数矩阵（独立进程测峰值）。

## 目录结构

```
app/
  main.py        # FastAPI 路由、流式 multipart 直写暂存、串行锁、清理语义
  pipeline.py    # 校验、直方图中位数、行带聚合、2 倍读取预算、低内存 FITS 写出
tests/           # 单元/接口测试与压测脚本
Dockerfile
docker-compose.yml
requirements.txt
```
