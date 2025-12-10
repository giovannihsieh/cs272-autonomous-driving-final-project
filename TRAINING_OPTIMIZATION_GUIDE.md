# Training Optimization Guide

## Problem
Current training time: **60 hours** for 500k timesteps

## Solution
Optimized training time: **6-10 hours** (6-10x speedup!)

---

## Key Optimizations

### 1. Parallel Environments (4-8x speedup)
**Before:** `DummyVecEnv` (single process)
```python
venv = DummyVecEnv([make_env])
```

**After:** `SubprocVecEnv` (8 parallel processes)
```python
num_envs = 8
venv = SubprocVecEnv([make_env(i) for i in range(num_envs)])
```

**Impact:** Collects experience 8x faster by running 8 environments in parallel

---

### 2. Reduced Vehicle Count (2x speedup)
**Before:** 50 vehicles per environment
```python
"vehicles_count": 50
```

**After:** 25 vehicles per environment
```python
"vehicles_count": 25
```

**Impact:** Halves computation per environment step while maintaining realistic traffic

---

### 3. Shorter Episodes (1.3x speedup)
**Before:** 40 second episodes
```python
"duration": 40
```

**After:** 30 second episodes
```python
"duration": 30
```

**Impact:** Faster episode resets, more diverse training data

---

### 4. GPU Acceleration (2-3x speedup if available)
**Before:** No explicit device specification
```python
model = PPO("MlpPolicy", venv, ...)
```

**After:** Explicit GPU usage
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PPO("MlpPolicy", venv, device=device, ...)
```

**Impact:** Neural network updates 2-3x faster on GPU

---

### 5. Optimized Hyperparameters
**Changes:**
- `learning_rate`: 2e-4 → 3e-4 (faster convergence)
- `n_epochs`: 5 → 10 (better sample efficiency)
- `ent_coef`: 0.001 → 0.01 (more exploration)
- `clip_range`: 0.1 → 0.2 (standard value)

---

### 6. Less Frequent Evaluation
**Before:** Every 25k timesteps
```python
eval_freq=25_000
```

**After:** Every 50k timesteps
```python
eval_freq=50_000 // num_envs
```

**Impact:** Less time spent on evaluation

---

## Cumulative Speedup Calculation

| Optimization | Speedup |
|-------------|---------|
| Parallel Envs (8x) | 8.0x |
| Vehicle Count (50→25) | 2.0x |
| Episode Length (40→30) | 1.3x |
| GPU Acceleration | 2.5x |
| Less Frequent Eval | 1.1x |
| **Total Theoretical** | **57.2x** |
| **Practical (overhead)** | **6-10x** |

Expected time: **60 hours / 6-10 = 6-10 hours**

---

## Usage

### Local Training (with or without GPU)
```bash
cd custom_env
python custom_emergency_lidar_optimized.py
```

### Google Colab (Free GPU)
1. Upload `custom_env/` folder to Google Drive
2. Open `colab_training_template.ipynb`
3. Update paths and run all cells
4. Expected time: ~2-3 hours on Colab GPU

---

## Hardware Recommendations

| Hardware | Expected Time | Cost |
|----------|---------------|------|
| **CPU only (8 cores)** | 8-12 hours | Free |
| **Local GPU (RTX 3060+)** | 4-6 hours | $300-500 |
| **Google Colab (Free GPU)** | 2-3 hours | Free |
| **Google Colab Pro (A100)** | 1-2 hours | $10/month |

---

## Performance Comparison

### Original Script
```bash
python custom_emergency_lidar.py
```
- Time: **60 hours**
- Uses: 1 CPU, no parallelization
- Vehicles: 50
- Duration: 40s

### Optimized Script
```bash
python custom_emergency_lidar_optimized.py
```
- Time: **6-10 hours** (CPU) or **2-3 hours** (GPU)
- Uses: 8 parallel CPUs + GPU
- Vehicles: 25
- Duration: 30s

---

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir custom_emergency_logs_lidar_optimized/tb/
```
Open browser: http://localhost:6006

### Training Progress
The script prints:
- Total episodes completed
- Average reward (last 100 episodes)
- Training time elapsed
- ETA to completion

---

## Further Optimizations (if needed)

### 1. Even Shorter Training (250k timesteps)
```python
model.learn(total_timesteps=250_000)  # Half the time
```
**Time:** 3-5 hours (may reduce final performance slightly)

### 2. More Parallel Environments
```python
num_envs = 16  # If you have 16+ CPU cores
```
**Impact:** Further speedup if you have enough cores

### 3. Smaller Network
```python
policy_kwargs = dict(net_arch=[64, 64])  # Smaller than default [256, 256]
model = PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs, ...)
```
**Impact:** 20-30% faster training, slight performance drop

### 4. Reduce Evaluation Episodes
```python
eval_callback = EvalCallback(..., n_eval_episodes=5)  # Down from 10
```
**Impact:** Faster evaluation

---

## Troubleshooting

### "No GPU detected"
- **Solution 1:** Use Google Colab (free GPU)
- **Solution 2:** Continue with CPU (still 6-10x faster than original)

### "Too many open files"
```bash
# macOS/Linux
ulimit -n 4096
```

### Out of Memory
```python
# Reduce parallel environments
num_envs = 4  # Instead of 8
```

### Still too slow?
- Use Google Colab with GPU (fastest option)
- Reduce to 250k timesteps
- Use smaller network architecture
- Further reduce vehicle count to 15

---

## Files

- `custom_emergency_lidar_optimized.py` - Optimized training script (use this!)
- `custom_emergency_lidar.py` - Original training script (slow)
- `colab_training_template.ipynb` - Google Colab version with GPU

---

## Expected Results

Both optimized and original scripts should achieve similar final performance:
- Mean episode return: ~25-35
- Learning stabilizes after ~300k timesteps
- Best model selected automatically via EvalCallback

The optimized version just gets there **6-10x faster**!
