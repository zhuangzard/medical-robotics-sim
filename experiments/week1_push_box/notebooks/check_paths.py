"""
检查所有模型路径和文件
在 Colab 中运行：!python3 experiments/week1_push_box/notebooks/check_paths.py
"""
import os
import glob

print("="*60)
print("🔍 Checking Model Paths and Files")
print("="*60)

# Check models directory
models_dir = "./models"
print(f"\n📂 Models Directory: {models_dir}")
if os.path.exists(models_dir):
    print("  ✅ Directory exists")
    
    # List all files
    all_files = glob.glob(f"{models_dir}/**/*", recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    
    if all_files:
        print(f"\n  📄 Found {len(all_files)} files:")
        for f in sorted(all_files):
            size_kb = os.path.getsize(f) / 1024
            print(f"    - {f} ({size_kb:.1f} KB)")
    else:
        print("  ⚠️  No files found")
else:
    print("  ❌ Directory not found")

# Check expected model paths
print(f"\n📋 Expected Model Paths (by eval.py):")
expected = [
    "./models/pure_ppo_final.zip",
    "./models/gns_final.zip",
    "./models/physrobot_final.zip"
]

for path in expected:
    exists = os.path.exists(path)
    symbol = "✅" if exists else "❌"
    print(f"  {symbol} {path}")

# Check old paths (in case user has old training)
print(f"\n🔧 Old Model Paths (deprecated):")
old_paths = [
    "./models/ppo/ppo_baseline.zip",
    "./models/gns/gns_baseline.zip",
    "./models/physrobot/physrobot_baseline.zip"
]

for path in old_paths:
    exists = os.path.exists(path)
    if exists:
        print(f"  ⚠️  {path} (should rename to pure_ppo_final.zip)")

# Summary
print("\n" + "="*60)
print("📊 Summary")
print("="*60)

expected_count = sum(1 for p in expected if os.path.exists(p))
print(f"Expected models found: {expected_count}/3")

if expected_count == 0:
    print("\n💡 Suggestion:")
    print("   1. Make sure training completed successfully")
    print("   2. Check training cell saves to correct path:")
    print("      model.save('./models/pure_ppo_final')")
elif expected_count < 3:
    print("\n💡 Suggestion:")
    print(f"   Only PPO trained. GNS and PhysRobot require separate training.")
else:
    print("\n✅ All models ready for OOD testing!")

print("="*60)
